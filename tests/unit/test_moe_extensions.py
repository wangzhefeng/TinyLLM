"""MoE 扩展测试：负载均衡 aux loss + shared expert（Phase 2 预训练域）。"""
import unittest
from types import SimpleNamespace

import torch

from layers.moe import MoEFeedForward


def _moe_cfg(num_shared=0, num_experts=4, top_k=2, vocab=32):
    return SimpleNamespace(
        embed_dim=32, moe_intermediate_size=64, num_experts=num_experts,
        num_experts_per_tok=top_k, num_shared_experts=num_shared,
        dtype=torch.float32,
    )


class TestLoadBalancingLoss(unittest.TestCase):
    def test_balanced_lower_than_skewed(self):
        """路由均衡时 aux loss 明显小于偏斜路由。"""
        n_tokens, n_experts, top_k = 100, 4, 2
        # 均衡：每 token 选中 {0,1}/{2,3} 交替，概率 0.5/0.5
        idx_bal = torch.tensor([[0, 1], [2, 3]] * (n_tokens // 2))
        probs_bal = torch.full((n_tokens, top_k), 0.5)
        # 偏斜：全部 token 都选中 {0,1}
        idx_skew = torch.tensor([[0, 1]] * n_tokens)
        probs_skew = torch.full((n_tokens, top_k), 0.5)

        loss_bal = MoEFeedForward.load_balancing_loss(probs_bal, idx_bal, n_experts)
        loss_skew = MoEFeedForward.load_balancing_loss(probs_skew, idx_skew, n_experts)
        # 精确值：均衡 f=[.5]*4, P=[.25]*4 → N·Σf·P = 4·0.5 = 2；
        # 偏斜 f=[1,1,0,0], P=[.5,.5,0,0] → 4·1.0 = 4
        self.assertAlmostEqual(float(loss_bal), 2.0, places=5)
        self.assertAlmostEqual(float(loss_skew), 4.0, places=5)

    def test_perfectly_balanced_near_one(self):
        """均匀路由（每专家 f=P=1/N·top_k 权重和）时 loss 接近 1 量级。"""
        n_tokens, n_experts, top_k = 400, 4, 1
        idx = torch.arange(n_tokens) % n_experts
        probs = torch.full((n_tokens, 1), 1.0)
        loss = MoEFeedForward.load_balancing_loss(probs, idx.unsqueeze(1), n_experts)
        self.assertAlmostEqual(float(loss), 1.0, places=5)


class TestSharedExpert(unittest.TestCase):
    def test_forward_shape_and_aux(self):
        torch.manual_seed(0)
        moe = MoEFeedForward(_moe_cfg(num_shared=1))
        x = torch.randn(2, 5, 32)
        out, aux = moe(x, return_aux_loss=True)
        self.assertEqual(out.shape, (2, 5, 32))
        self.assertTrue(torch.isfinite(aux))
        self.assertTrue(moe.last_aux_loss is not None)

    def test_shared_expert_affects_output(self):
        """shared expert 参与计算：置零其权重应改变输出。"""
        torch.manual_seed(0)
        moe = MoEFeedForward(_moe_cfg(num_shared=1)).eval()
        x = torch.randn(1, 4, 32)
        with torch.no_grad():
            out1 = moe(x)
            saved = [w.clone() for m in (moe.shared_fc1, moe.shared_fc2, moe.shared_fc3)
                     for w in m.parameters()]
            for m in (moe.shared_fc1, moe.shared_fc2, moe.shared_fc3):
                for p in m.parameters():
                    p.zero_()
            out2 = moe(x)
        self.assertFalse(torch.allclose(out1, out2))

    def test_no_shared_back_compat(self):
        """不配置 num_shared_experts 时行为与旧版一致（无 shared 模块）。"""
        cfg = _moe_cfg()
        moe = MoEFeedForward(cfg)
        self.assertEqual(moe.num_shared_experts, 0)
        self.assertFalse(hasattr(moe, "shared_fc1"))
        out = moe(torch.randn(1, 4, 32))
        self.assertEqual(out.shape, (1, 4, 32))


if __name__ == "__main__":
    unittest.main()
