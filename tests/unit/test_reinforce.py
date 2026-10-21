"""REINFORCE / REINFORCE-with-baseline 与 reasoning-trace 蒸馏测试（reasoning ch06/ch08）。"""
import unittest
from types import SimpleNamespace

import torch

from post_training.reasoning.training.reinforce import (
    compute_reinforce_loss,
    reinforce_step,
)
from post_training.reasoning.training.distill_reasoning import (
    generate_reasoning_traces,
    reasoning_sft_loss,
    distill_sft_step,
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


class TestReinforceLoss(unittest.TestCase):
    def test_plain_reinforce_direction(self):
        """无 baseline：高 reward 序列的正 logprob 被放大（loss 为负）。"""
        logp = torch.tensor([-1.0, -2.0])
        rewards = torch.tensor([1.0, 0.0])
        loss = compute_reinforce_loss(logp, rewards, baseline=None)
        self.assertAlmostEqual(float(loss), -(-1.0 * 1.0 + -2.0 * 0.0) / 2)

    def test_mean_baseline_shifts_advantage(self):
        """批均值 baseline：advantage 均值为 0，常数 logprob 下 loss=0。"""
        logp = torch.tensor([-1.0, -1.0])
        rewards = torch.tensor([1.0, 0.0])
        loss = compute_reinforce_loss(logp, rewards, baseline="mean")
        self.assertAlmostEqual(float(loss), 0.0, places=6)

    def test_baseline_reduces_variance_motivation(self):
        """同一 reward 集合下，baseline 版的 |loss| 更小（降方差的直观体现）。"""
        torch.manual_seed(0)
        logp = -torch.rand(4)
        rewards = torch.tensor([10.0, 10.0, 0.0, 0.0])
        plain = compute_reinforce_loss(logp, rewards, baseline=None)
        with_b = compute_reinforce_loss(logp, rewards, baseline="mean")
        self.assertLess(abs(float(with_b)), abs(float(plain)))


class TestReinforceStep(unittest.TestCase):
    def test_reward_increases_over_steps(self):
        """与 GRPO 同款玩具任务：训练后高分 token 概率质量上升。"""
        torch.manual_seed(11)
        model = _tiny_gpt2()
        x = torch.randint(0, 32, (1, 5))

        def high_token_mass(m):
            with torch.no_grad():
                probs = torch.softmax(m(x)[:, -1, :], dim=-1)
                return float(probs[0, 17:].sum())

        before = high_token_mass(model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)
        reward_fn = lambda ids_row: 1.0 if int(ids_row[-1].item()) > 16 else 0.0
        for _ in range(10):
            reinforce_step(
                model, optimizer, x, reward_fn, num_samples=6,
                max_new_tokens=4, context_length=48, temperature=1.0,
                baseline="mean",
            )
        after = high_token_mass(model)
        self.assertGreater(after, before)

    def test_scalar_baseline_accepted(self):
        model = _tiny_gpt2()
        x = torch.randint(0, 32, (1, 4))
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        res = reinforce_step(model, optimizer, x, lambda r: 1.0,
                             num_samples=2, max_new_tokens=3,
                             context_length=48, baseline=0.5)
        self.assertAlmostEqual(res["baseline"], 0.5)


class TestReasoningTraces(unittest.TestCase):
    def test_verifier_filters_traces(self):
        """verifier 只保留 reward>0 的轨迹。"""
        model = _tiny_gpt2().eval()
        x = torch.randint(0, 32, (1, 4))
        # 一半 token 判 True 一半判 False 的确定性 verifier：按首 token 奇偶
        def verifier(row):
            return int(row[-1].item()) % 2 == 0
        torch.manual_seed(7)
        traces = generate_reasoning_traces(
            model, x, verifier_fn=verifier, num_samples=6,
            max_new_tokens=4, context_length=48, temperature=1.0,
        )
        # 采样的最后 token 奇偶混合时应有保留也有丢弃；至少结构合法
        for tr in traces:
            self.assertEqual(tr["prompt_ids"].shape, x.shape)
            self.assertTrue(tr["completion_ids"].shape[1] >= 1)
            # 轨迹末 token（生成部分的最后一个）必须通过 verifier
            self.assertEqual(int(tr["completion_ids"][0, -1].item()) % 2, 0)

    def test_no_verifier_keeps_all(self):
        model = _tiny_gpt2().eval()
        x = torch.randint(0, 32, (1, 4))
        traces = generate_reasoning_traces(
            model, x, verifier_fn=None, num_samples=3,
            max_new_tokens=4, context_length=48, temperature=1.0,
        )
        self.assertEqual(len(traces), 3)


class TestReasoningSFT(unittest.TestCase):
    def test_loss_matches_manual_ce_on_completion(self):
        """loss 数值 = 手工对 completion 部分的 CE（prompt 不计损）。"""
        torch.manual_seed(0)
        model = _tiny_gpt2().eval()
        prompt = torch.randint(0, 32, (1, 4))
        completion = torch.randint(0, 32, (1, 5))
        loss = reasoning_sft_loss(model, prompt, completion)
        # 手工计算
        with torch.no_grad():
            seq = torch.cat([prompt, completion], dim=1)
            logits = model(seq)[:, :-1, :]
            targets = seq[:, 1:]
            import torch.nn.functional as F
            manual = F.cross_entropy(
                logits[:, -5:, :].reshape(-1, 32), targets[:, -5:].reshape(-1))
        self.assertAlmostEqual(float(loss), float(manual), places=5)

    def test_distill_step_overfits_single_trace(self):
        """学生在同一条教师轨迹上反复 SFT，损失应下降。"""
        torch.manual_seed(2)
        student = _tiny_gpt2()
        prompt = torch.randint(0, 32, (1, 4))
        completion = torch.randint(0, 32, (1, 6))
        optimizer = torch.optim.AdamW(student.parameters(), lr=3e-3)
        first = distill_sft_step(student, optimizer, prompt, completion)["loss"]
        last = first
        for _ in range(30):
            last = distill_sft_step(student, optimizer, prompt, completion)["loss"]
        self.assertLess(last, first)


if __name__ == "__main__":
    unittest.main()
