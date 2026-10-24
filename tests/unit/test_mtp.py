"""DeepSeek-V3 MTP（Multi-Token Prediction，多 token 预测）模块测试。

覆盖：
1. MTPHead 前向形状（logits / h_new）
2. deepseek_mtp_loss 有限且可反向传播
3. 参数共享：shared_embedding/norm/out_head 与主模型为同一对象，
   attach_mtp 后 named_parameters 去重（共享参数只计一次）
4. cache 路径：两段增量前向不报错（复用 DeepSeekBlock 的 MLA cache 语义）
"""
import unittest

import torch

from models.deepseek.model import DeepSeekModel
from models.deepseek.mtp import (
    MTPHead,
    deepseek_mtp_loss,
    deepseek_mtp_joint_loss,
    deepseek_mtp_train_step,
)
from utils.args_tools import DotDict

# 照抄 models/deepseek/model.py main() 的 tiny 教学配置
DEEPSEEK_TINY_CONFIG = {
    "vocab_size": 1000,
    "context_length": 256,
    "embed_dim": 256,
    "n_layers": 4,
    "n_heads": 8,
    "kv_lora_rank": 64,           # V3: 512
    "q_lora_rank": 128,           # V3: 1536
    "qk_nope_head_dim": 24,       # V3: 128
    "qk_rope_head_dim": 8,        # V3: 64
    "hidden_dim": 512,            # FFN 中间维（FeedForwardSiLU 回退键）
    "rope_base": 10_000.0,
    "dtype": torch.bfloat16,
}


def _cfg():
    return DotDict(DEEPSEEK_TINY_CONFIG)


class TestMTPHeadForward(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)
        self.cfg = _cfg()
        self.model = DeepSeekModel(self.cfg).eval()

    def test_forward_shapes(self):
        head = MTPHead(self.cfg, self.model.tok_embed,
                       self.model.final_norm, self.model.out_head)
        ids = torch.randint(0, self.cfg.vocab_size, (2, 8))
        h_prev = torch.randn(2, 8, self.cfg.embed_dim, dtype=self.cfg.dtype)
        with torch.no_grad():
            logits, h_new = head(ids, h_prev)
        self.assertEqual(logits.shape, (2, 8, self.cfg.vocab_size))
        self.assertEqual(h_new.shape, (2, 8, self.cfg.embed_dim))

    def test_forward_default_norm_head(self):
        """shared_norm / shared_out_head 缺省时自建，前向形状不变。"""
        head = MTPHead(self.cfg, self.model.tok_embed)
        ids = torch.randint(0, self.cfg.vocab_size, (2, 8))
        h_prev = torch.randn(2, 8, self.cfg.embed_dim, dtype=self.cfg.dtype)
        with torch.no_grad():
            logits, h_new = head(ids, h_prev)
        self.assertEqual(logits.shape, (2, 8, self.cfg.vocab_size))
        self.assertEqual(h_new.shape, (2, 8, self.cfg.embed_dim))


class TestMTPLoss(unittest.TestCase):
    def test_loss_finite_and_backward(self):
        torch.manual_seed(123)
        cfg = _cfg()
        model = DeepSeekModel(cfg)
        head = MTPHead(cfg, model.tok_embed)
        ids = torch.randint(0, cfg.vocab_size, (2, 8))
        h_prev = torch.randn(2, 8, cfg.embed_dim, dtype=cfg.dtype)
        target = torch.randint(0, cfg.vocab_size, (2, 8))

        logits, _ = head(ids, h_prev)
        loss = deepseek_mtp_loss(logits, target)
        self.assertTrue(torch.isfinite(loss).item())
        loss.backward()
        grad = head.projection.weight.grad
        self.assertIsNotNone(grad)
        self.assertTrue(torch.isfinite(grad.float()).all().item())


class TestMTPSharing(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)
        self.cfg = _cfg()
        self.model = DeepSeekModel(self.cfg).eval()

    def test_shared_modules_identity(self):
        head = MTPHead(self.cfg, self.model.tok_embed,
                       self.model.final_norm, self.model.out_head)
        self.assertIs(head.shared_embedding, self.model.tok_embed)
        self.assertIs(head.shared_norm, self.model.final_norm)
        self.assertIs(head.shared_out_head, self.model.out_head)

    def test_attach_mtp_param_dedup(self):
        """attach_mtp(2) 后 named_parameters 去重数量 = 手动预期（共享只计一次）。"""
        base_ids = {id(p) for p in self.model.parameters()}
        heads = self.model.attach_mtp(2)
        self.assertEqual(len(heads), 2)
        self.assertIsInstance(heads, torch.nn.ModuleList)

        # 手动预期：主模型参数 + 每个 head 独有参数（combine_norm/projection/trf_block）
        expected_ids = set(base_ids)
        for head in heads:
            self.assertIs(head.shared_embedding, self.model.tok_embed)
            for mod in (head.combine_norm, head.projection, head.trf_block):
                expected_ids.update(id(p) for p in mod.parameters())

        named = list(self.model.named_parameters())
        # named_parameters 默认去重，共享参数只出现一次
        self.assertEqual(len(named), len(expected_ids))
        self.assertEqual({id(p) for _, p in named}, expected_ids)


class TestMTPCache(unittest.TestCase):
    def test_incremental_two_pass(self):
        """连续两次 forward（第二次传 cache + start_pos）不报错。"""
        torch.manual_seed(123)
        cfg = _cfg()
        model = DeepSeekModel(cfg).eval()
        head = MTPHead(cfg, model.tok_embed, model.final_norm, model.out_head).eval()

        ids1 = torch.randint(0, cfg.vocab_size, (2, 4))
        h1 = torch.randn(2, 4, cfg.embed_dim, dtype=cfg.dtype)
        ids2 = torch.randint(0, cfg.vocab_size, (2, 3))
        h2 = torch.randn(2, 3, cfg.embed_dim, dtype=cfg.dtype)

        with torch.no_grad():
            # 第一段：用 trf_block 跑出 MLA 的 (c_kv, k_rope) cache
            x1 = head.projection(head.combine_norm(
                torch.cat([h1, head.shared_embedding(ids1)], dim=-1)
            ))
            _, cache = head.trf_block(x1, start_pos=0)

            # 第二段：带 cache 增量前向
            logits2, h_new2 = head(ids2, h2, start_pos=4, cache=cache)
        self.assertEqual(logits2.shape, (2, 3, cfg.vocab_size))
        self.assertEqual(h_new2.shape, (2, 3, cfg.embed_dim))


class TestMTPJointLoss(unittest.TestCase):
    """MTP 训练编排：联合损失 + 训练步封装（2026-10-24 接线）。"""

    def setUp(self):
        torch.manual_seed(123)
        self.cfg = _cfg()
        self.model = DeepSeekModel(self.cfg)
        self.model.attach_mtp(2)

    def test_joint_loss_parts_consistent(self):
        ids = torch.randint(0, self.cfg.vocab_size, (2, 12))
        total, main_loss, mtp_losses = deepseek_mtp_joint_loss(
            self.model, ids, mtp_weight=0.5, return_parts=True)
        self.assertEqual(mtp_losses.shape, (2,))
        torch.testing.assert_close(total, main_loss + 0.5 * mtp_losses.mean())
        self.assertTrue(torch.isfinite(total).item())

    def test_weight_zero_equals_main(self):
        ids = torch.randint(0, self.cfg.vocab_size, (2, 12))
        total, main_loss, _ = deepseek_mtp_joint_loss(
            self.model, ids, mtp_weight=0.0, return_parts=True)
        torch.testing.assert_close(total, main_loss)

    def test_backward_reaches_backbone_and_heads(self):
        ids = torch.randint(0, self.cfg.vocab_size, (2, 12))
        deepseek_mtp_joint_loss(self.model, ids).backward()
        self.assertIsNotNone(self.model.tok_embed.weight.grad)
        self.assertIsNotNone(self.model.mtp_heads[0].projection.weight.grad)
        self.assertIsNotNone(self.model.mtp_heads[1].projection.weight.grad)

    def test_short_sequence_raises(self):
        ids = torch.randint(0, self.cfg.vocab_size, (2, 4))   # 需 >= n_mtp+3 = 5
        with self.assertRaises(ValueError):
            deepseek_mtp_joint_loss(self.model, ids)

    def test_hidden_states_optin(self):
        """forward 默认返回 logits；opt-in 返回 (logits, hidden)。"""
        ids = torch.randint(0, self.cfg.vocab_size, (2, 6))
        with torch.no_grad():
            out = self.model(ids)
            logits, hidden = self.model(ids, output_hidden_states=True)
        self.assertIsInstance(out, torch.Tensor)
        self.assertEqual(logits.shape, (2, 6, self.cfg.vocab_size))
        self.assertEqual(hidden.shape, (2, 6, self.cfg.embed_dim))

    def test_train_step_runs(self):
        optimizer = torch.optim.SGD(
            [p for p in self.model.parameters() if p.requires_grad], lr=0.01)
        ids = torch.randint(0, self.cfg.vocab_size, (2, 12))
        total1, main1, mtp1 = deepseek_mtp_train_step(self.model, optimizer, ids)
        total2, main2, mtp2 = deepseek_mtp_train_step(self.model, optimizer, ids)
        for v in (total1, main1, total2, main2):
            self.assertTrue(torch.isfinite(torch.tensor(v)).item())
        self.assertEqual(mtp1.shape, (2,))


if __name__ == "__main__":
    unittest.main()
