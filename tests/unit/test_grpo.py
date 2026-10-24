"""GRPO from scratch 测试（Phase 2 后训练域，reasoning ch06/07）。"""
import unittest
from types import SimpleNamespace

import torch

from post_training.rl.core import (
    completion_mask,
    compute_grpo_loss,
    compute_group_advantages,
    generate_group_samples,
    grpo_update_epoch,
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
        samples, old_logprobs, comp_lens = generate_group_samples(
            model, x, group_size=4, max_new_tokens=6, context_length=48,
            temperature=1.0,
        )
        self.assertEqual(samples.shape[0], 4)
        self.assertEqual(old_logprobs.shape[0], 4)
        self.assertEqual(old_logprobs.shape[1], samples.shape[1] - 5)
        self.assertTrue(torch.isfinite(old_logprobs).all())
        # completion 长度有效：1..max_new_tokens，且等于最长者决定 padding 宽度
        self.assertTrue(((comp_lens >= 1) & (comp_lens <= 6)).all())
        self.assertEqual(int(comp_lens.max()), samples.shape[1] - 5)


class TestGRPOLoss(unittest.TestCase):
    def test_zero_ratio_is_baseline(self):
        """新策略=旧策略时 ratio=1，loss = -mean(clip(min))，方向由优势决定。"""
        model = _tiny_gpt2().eval()
        x = torch.randint(0, 32, (1, 5))
        samples, old_logprobs, comp_lens = generate_group_samples(
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
        samples, old_logprobs, comp_lens = generate_group_samples(
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


class TestCompletionMask(unittest.TestCase):
    def test_mask_shape_and_values(self):
        lens = torch.tensor([4, 2])
        m = completion_mask(lens, max_len=4)
        self.assertEqual(m.shape, (2, 4))
        self.assertTrue(m[0].all())
        self.assertEqual(m[1].tolist(), [True, True, False, False])

    def test_masked_loss_excludes_padding(self):
        """提供 completion_lengths 时，padding 位不进入 loss：
        手工声明第二条样本只有前 2 个 token 有效，篡改其 padding 位的
        old_logprob 使 ratio 爆炸——masked loss 应与未篡改完全一致，
        而不 mask（None）时 loss 被污染。"""
        model = _tiny_gpt2().eval()
        x = torch.randint(0, 32, (1, 5))
        samples, old_logprobs, _ = generate_group_samples(
            model, x, group_size=2, max_new_tokens=4, context_length=48,
            temperature=1.0,
        )
        # 模拟第二条提前 EOS：真实 completion 长度为 2，其余为 padding
        comp_lens = torch.tensor([4, 2])
        adv = torch.ones(2)
        old_tampered = old_logprobs.clone()
        mask = completion_mask(comp_lens, old_logprobs.shape[1])
        old_tampered[~mask] = old_tampered[~mask] - 50.0   # padding 位 ratio 爆炸
        # masked：padding 被完全排除，篡改无影响
        loss_masked = compute_grpo_loss(model, samples, old_tampered, adv,
                                        prompt_len=5, clip_eps=0.2,
                                        completion_lengths=comp_lens)
        loss_ref = compute_grpo_loss(model, samples, old_logprobs, adv,
                                     prompt_len=5, clip_eps=0.2,
                                     completion_lengths=comp_lens)
        self.assertAlmostEqual(float(loss_masked), float(loss_ref), places=4)
        # unmasked：padding 位 ratio 被 clip 到 1+eps（=1.2），与有效位（ratio≈1）
        # 的贡献不同 → loss 被污染
        loss_unmasked = compute_grpo_loss(model, samples, old_tampered, adv,
                                          prompt_len=5, clip_eps=0.2,
                                          completion_lengths=None)
        self.assertGreater(abs(float(loss_unmasked) - float(loss_ref)), 1e-3)

    def test_eos_gives_short_completion(self):
        """传 eos_token_id 且采样命中 EOS 时，completion_lengths < max_new_tokens。"""
        model = _tiny_gpt2().eval()
        x = torch.randint(0, 32, (1, 3))
        # 用 argmax 必中的方式：temperature=0（贪心）+ 把 EOS 设为当前首选 token
        with torch.no_grad():
            first_token = int(torch.argmax(model(x)[:, -1, :], dim=-1).item())
        samples, old_logprobs, comp_lens = generate_group_samples(
            model, x, group_size=3, max_new_tokens=8, context_length=48,
            temperature=0.0, eos_token_id=first_token,
        )
        # 贪心首 token 即 EOS → 每条只生成 1 个 token
        self.assertTrue(bool((comp_lens == 1).all()))


class TestGRPOUpdateEpoch(unittest.TestCase):
    def test_epoch_runs_and_reports(self):
        """多 prompt 梯度累积更新：跑通并返回批指标。"""
        torch.manual_seed(5)
        model = _tiny_gpt2()
        prompts = [torch.randint(0, 32, (1, 4)) for _ in range(3)]
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        reward_fn = lambda ids_row: 1.0 if int(ids_row[-1].item()) > 16 else 0.0
        res = grpo_update_epoch(
            model, optimizer, prompts, reward_fn,
            group_size=4, max_new_tokens=3, context_length=48, temperature=1.0,
        )
        self.assertEqual(res["num_prompts"], 3)
        self.assertEqual(len(res["rewards"]), 12)
        self.assertTrue(torch.isfinite(torch.tensor(res["loss"])))


if __name__ == "__main__":
    unittest.main()
