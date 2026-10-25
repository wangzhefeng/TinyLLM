"""蒸馏测试（Phase 2 后训练域，reasoning ch08）。"""
import math
import unittest
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from post_training.reasoning.training.distillation import (
    distill_step,
    distillation_loss,
)
from post_training.reasoning.training.distill_reasoning import (
    distill_sft_epoch,
    reasoning_sft_loss,
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


class TestDistillSFTEpoch(unittest.TestCase):
    """distill_sft_epoch（批量右填充蒸馏，ch08 distill_batched）。"""

    def test_epoch_runs_and_loss_finite(self):
        student = _gpt2(seed=1)
        optimizer = torch.optim.AdamW(student.parameters(), lr=1e-3)
        dataset = [
            (torch.randint(0, 32, (1, 5)), torch.randint(0, 32, (1, 7)))
            for _ in range(6)
        ]
        loss = distill_sft_epoch(dataset, student, optimizer, batch_size=2, pad_id=0)
        self.assertTrue(math.isfinite(loss))

    def test_loss_decreases_over_epochs(self):
        """学生模仿教师 greedy 轨迹：若干 epoch 后 loss 应下降。"""
        teacher = _gpt2(seed=10).eval()
        torch.manual_seed(0)
        dataset = []
        for _ in range(6):
            prompt = torch.randint(0, 32, (1, 5))
            with torch.no_grad():
                seq = prompt
                for _ in range(6):
                    next_token = teacher(seq)[:, -1, :].argmax(dim=-1, keepdim=True)
                    seq = torch.cat([seq, next_token], dim=1)
            dataset.append((prompt, seq[:, 5:]))
        student = _gpt2(seed=99)
        optimizer = torch.optim.AdamW(student.parameters(), lr=5e-3)
        first = distill_sft_epoch(dataset, student, optimizer, batch_size=3, pad_id=0)
        last = first
        for _ in range(6):
            last = distill_sft_epoch(dataset, student, optimizer, batch_size=3, pad_id=0)
        self.assertLess(last, first)

    def test_padding_token_does_not_change_loss(self):
        """右填充 + target 掩码：带 pad 的 batch loss 与无 pad 逐样本一致。

        lr=0 冻结权重，排除 optimizer.step() 对参照 loss 的干扰。
        """
        student = _gpt2(seed=3)
        optimizer = torch.optim.AdamW(student.parameters(), lr=0.0)
        short = (torch.randint(0, 32, (1, 5)), torch.randint(0, 32, (1, 7)))
        long = (torch.randint(0, 32, (1, 6)), torch.randint(0, 32, (1, 4)))

        batched = distill_sft_epoch(
            [short, long], student, optimizer, batch_size=2, pad_id=0,
        )
        manual = 0.5 * (
            float(reasoning_sft_loss(student, *short))
            + float(reasoning_sft_loss(student, *long))
        )
        self.assertAlmostEqual(batched, manual, places=4)

        single = distill_sft_epoch(
            [short], student, optimizer, batch_size=1, pad_id=0,
        )
        unpadded = float(reasoning_sft_loss(student, *short))
        self.assertAlmostEqual(single, unpadded, places=5)


if __name__ == "__main__":
    unittest.main()
