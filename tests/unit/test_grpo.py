"""GRPO from scratch 测试（Phase 2 后训练域，reasoning ch06/07）。"""
import unittest
from types import SimpleNamespace

import torch

from post_training.reasoning.training.reinforcement_learning import (
    compute_grpo_loss,
    compute_group_advantages,
    generate_group_samples,
    grpo_update_step,
)
from models.gpt2.gpt2_124M import Model


def _tiny_gpt2(vocab=32):
    torch.manual_seed(3)
    cfg = SimpleNamespace(
        vocab_size=vocab, context_length=48, embed_dim=32, d_ff=64,
        n_heads=4, n_layers=2, dropout=0.0, qkv_bias=True,
        dtype=torch.float32, is_train=False,
    )
    return Model(cfg)


class TestGroupAdvantages(unittest.TestCase):
    def test_mean_zero_and_sign(self):
        r = torch.tensor([1.0, 0.0, 0.0, 0.0])
        adv = compute_group_advantages(r)
        self.assertAlmostEqual(float(adv.mean()), 0.0, places=5)
        self.assertGreater(float(adv[0]), 0)      # 高分者优势为正
        self.assertTrue((adv[1:] < 0).all())

    def test_constant_rewards_give_zero(self):
        adv = compute_group_advantages(torch.ones(4))
        self.assertTrue(torch.allclose(adv, torch.zeros(4), atol=1e-5))


class TestGenerateGroupSamples(unittest.TestCase):
    def test_shapes_and_old_logprobs(self):
        model = _tiny_gpt2().eval()
        x = torch.randint(0, 32, (1, 5))
        samples, old_logprobs = generate_group_samples(
            model, x, group_size=4, max_new_tokens=6, context_length=48,
            temperature=1.0,
        )
        self.assertEqual(samples.shape[0], 4)
        self.assertEqual(old_logprobs.shape[0], 4)
        self.assertEqual(old_logprobs.shape[1], samples.shape[1] - 5)
        self.assertTrue(torch.isfinite(old_logprobs).all())


class TestGRPOLoss(unittest.TestCase):
    def test_zero_ratio_is_baseline(self):
        """新策略=旧策略时 ratio=1，loss = -mean(clip(min))，方向由优势决定。"""
        model = _tiny_gpt2().eval()
        x = torch.randint(0, 32, (1, 5))
        samples, old_logprobs = generate_group_samples(
            model, x, group_size=2, max_new_tokens=4, context_length=48,
            temperature=1.0,
        )
        adv = compute_group_advantages(torch.tensor([1.0, 0.0]))
        loss = compute_grpo_loss(model, samples, old_logprobs, adv,
                                 prompt_len=5, clip_eps=0.2)
        # ratio=1 → loss = -adv 的均值（负号在 loss 中）
        expected = -adv.mean().item()
        self.assertAlmostEqual(float(loss), expected, places=4)

    def test_clipping_limits_large_ratio(self):
        """ratio 远超 1+eps 时，clip 后的优势项被截断。"""
        model = _tiny_gpt2().eval()
        x = torch.randint(0, 32, (1, 5))
        samples, old_logprobs = generate_group_samples(
            model, x, group_size=2, max_new_tokens=4, context_length=48,
            temperature=1.0,
        )
        adv = torch.ones(2) * 1.0
        # 人为把 old_logprobs 设得很小 → ratio 爆炸
        old_small = old_logprobs - 5.0
        loss = compute_grpo_loss(model, samples, old_small, adv,
                                 prompt_len=5, clip_eps=0.2)
        # 优势全正 + ratio 爆炸 → 完全走 clip 分支：loss = -(1+eps)·adv
        self.assertAlmostEqual(float(loss), -1.2, places=3)


class TestGRPOUpdateStep(unittest.TestCase):
    def test_reward_increases_over_steps(self):
        """稠密奖励玩具任务：reward = 最后 token > 16（命中率 ~50%，组内必有方差）。
        验证学习信号确定性：训练后 prompt 处对高分 token 的概率质量上升。"""
        torch.manual_seed(11)
        model = _tiny_gpt2()
        x = torch.randint(0, 32, (1, 5))

        def high_token_mass(m):
            with torch.no_grad():
                probs = torch.softmax(m(x)[:, -1, :], dim=-1)
                return float(probs[0, 17:].sum())   # token 17..31 的概率质量

        before = high_token_mass(model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)
        reward_fn = lambda ids_row: 1.0 if int(ids_row[-1].item()) > 16 else 0.0
        for _ in range(10):
            grpo_update_step(
                model, optimizer, x, reward_fn, group_size=6, max_new_tokens=4,
                context_length=48, temperature=1.0,
            )
        after = high_token_mass(model)
        self.assertGreater(after, before)


if __name__ == "__main__":
    unittest.main()
