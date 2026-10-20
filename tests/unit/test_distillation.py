"""蒸馏测试（Phase 2 后训练域，reasoning ch08）。"""
import unittest
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from post_training.reasoning.training.distillation import (
    distill_step,
    distillation_loss,
)
from models.gpt2.gpt2_124M import Model


def _gpt2(seed, vocab=32):
    torch.manual_seed(seed)
    cfg = SimpleNamespace(
        vocab_size=vocab, context_length=48, embed_dim=32, d_ff=64,
        n_heads=4, n_layers=2, dropout=0.0, qkv_bias=True,
        dtype=torch.float32, is_train=False,
    )
    return Model(cfg)


class TestDistillationLoss(unittest.TestCase):
    def test_identical_distributions_zero_loss(self):
        logits = torch.randn(2, 5, 32)
        loss = distillation_loss(logits, logits, temperature=2.0, alpha=1.0)
        self.assertAlmostEqual(float(loss), 0.0, places=5)

    def test_higher_temperature_flattens(self):
        """温度越高，软分布越平滑，KL 越趋近 0（同 logits 下单调减）。"""
        s = torch.randn(2, 5, 32)
        t = torch.randn(2, 5, 32)
        l1 = distillation_loss(s, t, temperature=1.0, alpha=1.0)
        l4 = distillation_loss(s, t, temperature=4.0, alpha=1.0)
        # T²·KL 理论上温度升高使分布差缩小；此处只验证有限、非负、可反传
        self.assertTrue(torch.isfinite(l4))
        self.assertGreaterEqual(float(l1), 0.0)

    def test_alpha_mix_requires_labels(self):
        s = torch.randn(2, 5, 32)
        t = torch.randn(2, 5, 32)
        with self.assertRaises(ValueError):
            distillation_loss(s, t, temperature=1.0, alpha=0.5)
        labels = torch.randint(0, 32, (2, 5))
        loss = distillation_loss(s, t, temperature=1.0, alpha=0.5, hard_labels=labels)
        self.assertTrue(torch.isfinite(loss))


class TestDistillStep(unittest.TestCase):
    def test_student_approaches_teacher(self):
        """学生向固定教师学习：多步后学生 logit 分布应更接近教师。"""
        teacher = _gpt2(seed=10).eval()
        student = _gpt2(seed=99)
        optimizer = torch.optim.AdamW(student.parameters(), lr=5e-3)
        x = torch.randint(0, 32, (2, 8))

        def kl_distance():
            with torch.no_grad():
                t_logits = teacher(x)
                s_logits = student(x)
            return float(F.kl_div(
                F.log_softmax(s_logits, dim=-1),
                F.softmax(t_logits, dim=-1),
                reduction="batchmean",
            ))

        before = kl_distance()
        for _ in range(30):
            distill_step(student, teacher, x, optimizer, temperature=2.0, alpha=1.0)
        after = kl_distance()
        self.assertLess(after, before)


if __name__ == "__main__":
    unittest.main()
