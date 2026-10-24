"""MoE eval/train 行为测试：噪声路由确定性、eval 不计算 aux loss（共享层正确性修复）。"""
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from layers.moe import MoEFeedForward, SparseMoE, _NoisyTopkRouter


def _sparse_cfg():
    return SimpleNamespace(
        embed_dim=8, d_ff=16, dropout=0.0, top_k=2, num_experts=4,
    )


def _moe_ffn_cfg(num_experts=4, top_k=2):
    return SimpleNamespace(
        embed_dim=32, moe_intermediate_size=64, num_experts=num_experts,
        num_experts_per_tok=top_k, dtype=torch.float32,
    )


class TestNoisyTopkRouterNoiseGating(unittest.TestCase):
    def test_eval_is_deterministic(self):
        """eval 模式：两次 forward 路由结果完全一致（不加噪声）。"""
        torch.manual_seed(0)
        router = _NoisyTopkRouter(_sparse_cfg()).eval()
        x = torch.randn(2, 3, 8)
        with torch.no_grad():
            out1, idx1 = router(x)
            out2, idx2 = router(x)
        torch.testing.assert_close(out1, out2)
        self.assertTrue(torch.equal(idx1, idx2))

    def test_train_adds_noise(self):
        """train 模式：Noisy Top-k Gating 注入随机噪声，两次结果一般不同。"""
        torch.manual_seed(0)
        router = _NoisyTopkRouter(_sparse_cfg()).train()
        x = torch.randn(2, 3, 8)
        torch.manual_seed(1)
        out1, _ = router(x)
        torch.manual_seed(2)
        out2, _ = router(x)
        self.assertFalse(torch.allclose(out1, out2))

    def test_eval_matches_noiseless_logits(self):
        """eval 路由等价于对无线性噪声的 logits 直接取 top-k softmax。"""
        torch.manual_seed(0)
        router = _NoisyTopkRouter(_sparse_cfg()).eval()
        x = torch.randn(2, 3, 8)
        with torch.no_grad():
            out, indices = router(x)
            logits = router.topk_route_linear(x)
            topk_logits, expected_idx = torch.topk(logits, router.top_k, dim=-1)
            sparse = torch.scatter(
                torch.full_like(logits, float("-inf")), -1, expected_idx, topk_logits
            )
            expected_out = torch.softmax(sparse, dim=-1)
        self.assertTrue(torch.equal(indices, expected_idx))
        torch.testing.assert_close(out, expected_out)


class TestSparseMoEEvalDeterminism(unittest.TestCase):
    def test_eval_forward_deterministic(self):
        """SparseMoE 在 eval 下两次 forward 输出一致。"""
        torch.manual_seed(0)
        moe = SparseMoE(_sparse_cfg()).eval()
        x = torch.randn(2, 3, 8)
        with torch.no_grad():
            out1 = moe(x)
            out2 = moe(x)
        torch.testing.assert_close(out1, out2)

    def test_train_output_shape(self):
        """train 模式输出形状不变（API 兼容）。"""
        torch.manual_seed(0)
        moe = SparseMoE(_sparse_cfg()).train()
        out = moe(torch.randn(2, 3, 8))
        self.assertEqual(out.shape, (2, 3, 8))


class TestMoEFeedForwardAuxLossMode(unittest.TestCase):
    def test_train_computes_aux_loss(self):
        """train 模式：计算并返回 aux loss，last_aux_loss 被设置。"""
        torch.manual_seed(0)
        moe = MoEFeedForward(_moe_ffn_cfg()).train()
        x = torch.randn(2, 5, 32)
        out, aux = moe(x, return_aux_loss=True)
        self.assertEqual(out.shape, (2, 5, 32))
        self.assertIsNotNone(aux)
        self.assertTrue(torch.isfinite(aux))
        self.assertIsNotNone(moe.last_aux_loss)

    def test_eval_skips_aux_loss(self):
        """eval 模式：不计算 aux loss，last_aux_loss 为 None，返回 (out, None)。"""
        torch.manual_seed(0)
        moe = MoEFeedForward(_moe_ffn_cfg()).eval()
        x = torch.randn(2, 5, 32)
        with mock.patch.object(
            MoEFeedForward, "load_balancing_loss",
            side_effect=AssertionError("eval 不应计算 aux loss"),
        ):
            with torch.no_grad():
                out, aux = moe(x, return_aux_loss=True)
        self.assertEqual(out.shape, (2, 5, 32))
        self.assertIsNone(aux)
        self.assertIsNone(moe.last_aux_loss)

    def test_eval_default_call_returns_tensor(self):
        """eval 默认调用仍只返回输出张量（向后兼容）。"""
        torch.manual_seed(0)
        moe = MoEFeedForward(_moe_ffn_cfg()).eval()
        with torch.no_grad():
            out = moe(torch.randn(1, 4, 32))
        self.assertEqual(out.shape, (1, 4, 32))
        self.assertIsNone(moe.last_aux_loss)

    def test_output_numerics_match_unique_expert_loop(self):
        """range 遍历与旧 unique+.item() 遍历数值一致：同权重同输入输出相同。"""
        torch.manual_seed(0)
        moe = MoEFeedForward(_moe_ffn_cfg()).eval()
        x = torch.randn(2, 5, 32)

        def reference_forward(x):
            scores = moe.gate(x)
            topk_scores, topk_indices = torch.topk(scores, moe.num_experts_per_tok, dim=-1)
            topk_probs = torch.softmax(topk_scores, dim=-1)
            batch, seq_len, _ = x.shape
            x_flat = x.reshape(batch * seq_len, -1)
            out_flat = torch.zeros(batch * seq_len, moe.embed_dim)
            idx_flat = topk_indices.reshape(-1, moe.num_experts_per_tok)
            probs_flat = topk_probs.reshape(-1, moe.num_experts_per_tok)
            for e in torch.unique(idx_flat):
                eid = int(e.item())
                mask = idx_flat == eid
                sel = mask.any(dim=-1).nonzero(as_tuple=False).squeeze(-1)
                if sel.numel() == 0:
                    continue
                ei = x_flat.index_select(0, sel)
                h = torch.nn.functional.silu(moe.fc1[eid](ei)) * moe.fc2[eid](ei)
                eo = moe.fc3[eid](h)
                slot = mask[sel].int().argmax(dim=-1, keepdim=True)
                p = torch.gather(probs_flat.index_select(0, sel), -1, slot).squeeze(-1)
                out_flat.index_add_(0, sel, eo * p.unsqueeze(-1))
            return out_flat.reshape(batch, seq_len, moe.embed_dim)

        with torch.no_grad():
            out = moe(x)
            ref = reference_forward(x)
        torch.testing.assert_close(out, ref)


if __name__ == "__main__":
    unittest.main()
