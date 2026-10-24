"""PPO（序列级 bandit 教学版）测试，对照 tests/unit/test_grpo.py 的玩具模型写法。

覆盖：
- ratio=1 基线（actor loss = -mean(A)）
- 大 ratio 被 clip 截断
- critic loss 在固定批次上随训练下降
- 学习信号确定性（高分 token 概率质量上升）
- completion mask 排除 padding
"""
import unittest
from types import SimpleNamespace

import torch

from post_training.preference.model_rlhf.ppo.main import (
    CriticValueHead,
    compute_advantages_ppo,
    compute_critic_loss,
    compute_ppo_actor_loss,
    ppo_update_step,
)
from post_training.rl.core import (
    completion_mask,
    generate_group_samples,
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


class TestPPOActorLoss(unittest.TestCase):
    def test_zero_ratio_is_baseline(self):
        """新策略=旧策略时 ratio=1，actor loss = -mean(A)（min(1·A, clip·A)=A）。"""
        model = _tiny_gpt2().eval()
        x = torch.randint(0, 32, (1, 5))
        samples, old_logprobs, comp_lens = generate_group_samples(
            model, x, group_size=2, max_new_tokens=4, context_length=48,
            temperature=1.0,
        )
        # 注入常数 baseline：A = r - 0.5
        rewards = torch.tensor([1.0, 0.0], device=samples.device)
        adv = compute_advantages_ppo(rewards, torch.full((2,), 0.5))
        loss = compute_ppo_actor_loss(model, samples, old_logprobs, adv,
                                      prompt_len=5, clip_eps=0.2,
                                      completion_lengths=comp_lens)
        self.assertAlmostEqual(float(loss), -float(adv.mean()), places=4)

    def test_clipping_limits_large_ratio(self):
        """ratio 远超 1+eps 时，clip 后的优势项被截断：loss = -(1+eps)·mean(A)。"""
        model = _tiny_gpt2().eval()
        x = torch.randint(0, 32, (1, 5))
        samples, old_logprobs, comp_lens = generate_group_samples(
            model, x, group_size=2, max_new_tokens=4, context_length=48,
            temperature=1.0,
        )
        adv = torch.ones(2)
        old_small = old_logprobs - 5.0   # ratio 爆炸
        loss = compute_ppo_actor_loss(model, samples, old_small, adv,
                                      prompt_len=5, clip_eps=0.2,
                                      completion_lengths=comp_lens)
        self.assertAlmostEqual(float(loss), -1.2, places=3)

    def test_masked_loss_excludes_padding(self):
        """completion mask：篡改 padding 位 old_logprob 不影响 masked loss。"""
        model = _tiny_gpt2().eval()
        x = torch.randint(0, 32, (1, 5))
        samples, old_logprobs, _ = generate_group_samples(
            model, x, group_size=2, max_new_tokens=4, context_length=48,
            temperature=1.0,
        )
        # 模拟第二条提前 EOS：真实 completion 长度 2，其余为 padding
        comp_lens = torch.tensor([4, 2], device=samples.device)
        adv = torch.ones(2, device=samples.device)
        old_tampered = old_logprobs.clone()
        mask = completion_mask(comp_lens, old_logprobs.shape[1])
        old_tampered[~mask] = old_tampered[~mask] - 50.0
        loss_masked = compute_ppo_actor_loss(model, samples, old_tampered, adv,
                                             prompt_len=5, clip_eps=0.2,
                                             completion_lengths=comp_lens)
        loss_ref = compute_ppo_actor_loss(model, samples, old_logprobs, adv,
                                          prompt_len=5, clip_eps=0.2,
                                          completion_lengths=comp_lens)
        self.assertAlmostEqual(float(loss_masked), float(loss_ref), places=4)
        loss_unmasked = compute_ppo_actor_loss(model, samples, old_tampered, adv,
                                               prompt_len=5, clip_eps=0.2,
                                               completion_lengths=None)
        self.assertGreater(abs(float(loss_unmasked) - float(loss_ref)), 1e-3)


class TestCritic(unittest.TestCase):
    def test_critic_loss_decreases_on_fixed_batch(self):
        """critic 损失：固定一批 (embedding, reward) 多次更新，MSE 单调趋降。"""
        torch.manual_seed(4)
        critic = CriticValueHead(embed_dim=32)
        opt = torch.optim.AdamW(critic.parameters(), lr=5e-2)
        embeds = torch.randn(8, 32)
        rewards = torch.tensor([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
        losses = []
        for _ in range(50):
            opt.zero_grad()
            values = critic(embeds)
            loss = compute_critic_loss(values, rewards)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        self.assertLess(losses[-1], losses[0])
        # 收敛后误差应显著小于初值
        self.assertLess(losses[-1], 0.2 * losses[0])

    def test_value_fn_constant_baseline_accepted(self):
        """value_fn 注入常数 baseline：advantages = r - 0.5，values 返回正确。"""
        model = _tiny_gpt2()
        x = torch.randint(0, 32, (1, 4))
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        res = ppo_update_step(
            model, optimizer, x, lambda r: 1.0,
            num_samples=2, max_new_tokens=3, context_length=48,
            value_fn=lambda prompt: torch.full((2,), 0.5),
        )
        self.assertTrue(torch.allclose(res["values"], torch.full((2,), 0.5)))
        self.assertTrue(torch.allclose(res["advantages"], torch.full((2,), 0.5)))

    def test_critic_and_value_fn_mutually_exclusive(self):
        model = _tiny_gpt2()
        x = torch.randint(0, 32, (1, 4))
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        critic = CriticValueHead(32)
        critic_opt = torch.optim.AdamW(critic.parameters(), lr=1e-2)
        with self.assertRaises(ValueError):
            ppo_update_step(
                model, optimizer, x, lambda r: 1.0,
                num_samples=2, max_new_tokens=3, context_length=48,
                critic=critic, critic_optimizer=critic_opt,
                value_fn=lambda p: torch.zeros(2),
            )

    def test_critic_requires_own_optimizer(self):
        model = _tiny_gpt2()
        x = torch.randint(0, 32, (1, 4))
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        with self.assertRaises(ValueError):
            ppo_update_step(
                model, optimizer, x, lambda r: 1.0,
                num_samples=2, max_new_tokens=3, context_length=48,
                critic=CriticValueHead(32), critic_optimizer=None,
            )


class TestPPOUpdateStep(unittest.TestCase):
    def test_reward_increases_over_steps(self):
        """玩具任务：reward = **首个**生成 token > 16（与探针位置对齐——
        探针测的是 prompt 位对高分 token 的概率质量，即首个生成位置；
        基线 0.5 让奖励样本上推、未奖励样本下压，信号确定性可见）。
        10 步后高分 token 概率质量上升。"""
        torch.manual_seed(11)
        model = _tiny_gpt2()
        x = torch.randint(0, 32, (1, 5))

        def high_token_mass(m):
            with torch.no_grad():
                probs = torch.softmax(m(x)[:, -1, :], dim=-1)
                return float(probs[0, 17:].sum())

        before = high_token_mass(model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)
        # 生成 4 个 token，row[-4] 即首个生成 token（与探针位置对齐）
        reward_fn = lambda ids_row: 1.0 if int(ids_row[-4].item()) > 16 else 0.0
        for _ in range(10):
            ppo_update_step(
                model, optimizer, x, reward_fn, num_samples=6,
                max_new_tokens=4, context_length=48, temperature=1.0,
                # 常数基线 0.5：奖励样本优势 +0.5（上推）、未奖励 -0.5（下压），
                # 比零基线（只上推）信号更强——正是 baseline 降方差的直观演示
                value_fn=lambda prompt: torch.full((6,), 0.5),
            )
        after = high_token_mass(model)
        self.assertGreater(after, before)

    def test_step_with_learned_critic(self):
        """带可学习 critic 的完整一步：actor/critic 损失有限，且 critic 参数确实更新。"""
        torch.manual_seed(11)
        model = _tiny_gpt2()
        x = torch.randint(0, 32, (1, 5))
        critic = CriticValueHead(embed_dim=32)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        critic_opt = torch.optim.AdamW(critic.parameters(), lr=1e-2)
        before = [p.detach().clone() for p in critic.parameters()]
        res = ppo_update_step(
            model, optimizer, x, lambda r: 1.0,
            num_samples=4, max_new_tokens=3, context_length=48,
            critic=critic, critic_optimizer=critic_opt,
        )
        self.assertTrue(torch.isfinite(torch.tensor(res["actor_loss"])))
        self.assertTrue(torch.isfinite(torch.tensor(res["critic_loss"])))
        # critic 参数确实被更新过（与初始快照不同）
        self.assertTrue(any(
            not torch.equal(p0, p1) for p0, p1 in zip(before, critic.parameters())
        ))


if __name__ == "__main__":
    unittest.main()
