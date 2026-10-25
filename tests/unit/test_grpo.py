"""GRPO from scratch 测试（Phase 2 后训练域，reasoning ch06/07）。"""
import csv
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from post_training.rl.batched import (
    generate_group_samples_batched,
    grpo_update_step_batched,
)
from post_training.rl.core import (
    completion_mask,
    compute_grpo_loss,
    compute_group_advantages,
    generate_group_samples,
    grpo_update_epoch,
    grpo_update_step,
    token_logprobs,
)
from post_training.rl.objectives import compute_clipped_policy_loss
from post_training.rl.rewards import correct_answer_reward, format_reward
from post_training.rl.tracking import (
    MetricsTracker,
    compute_clip_fraction,
    plot_metrics,
)
from models.gpt2.gpt2_124M import Model


def _tiny_gpt2(vocab=32, seed=3):
    torch.manual_seed(seed)
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

    def test_degenerate_rollout_truncated_on_nan(self):
        """logits 出现 NaN/inf 的退化 rollout 被提前截断而不是抛异常。"""

        class _NanAfterK(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.inner = _tiny_gpt2().eval()
                self.calls = 0

            def forward(self, x):
                out = self.inner(x)
                self.calls += 1
                if self.calls > 3:
                    out = out.clone()
                    out[:, -1, :] = float("nan")
                return out

        model = _NanAfterK().eval()
        x = torch.randint(0, 32, (1, 5))
        samples, old_logprobs, comp_lens = generate_group_samples(
            model, x, group_size=2, max_new_tokens=10, context_length=48,
            temperature=1.0,
        )
        # 第 2 条 rollout 首个 token 即 NaN 时截断长度为 0（下游
        # _masked_normalize 分母 clamp(min=1) 保证全退化组 loss=0 而非 NaN）
        self.assertTrue(((comp_lens >= 0) & (comp_lens <= 10)).all())
        self.assertTrue(torch.isfinite(samples).all())
        self.assertTrue(torch.isfinite(old_logprobs).all())


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


class TestClippedPolicyLossExtensions(unittest.TestCase):
    """objectives.compute_clipped_policy_loss 的 ch07 扩展项。"""

    def _sample_group(self, model):
        x = torch.randint(0, 32, (1, 5))
        return generate_group_samples(
            model, x, group_size=2, max_new_tokens=4, context_length=48,
            temperature=1.0,
        )

    def test_clip_eps_high_only_widens_upper_bound(self):
        """大 ratio+正优势走上界：clip_eps_high=0.5 时 loss=-(1+0.5)；
        小 ratio+负优势走下界：loss 与 clip_eps_high 无关（下界仍 1-clip_eps）。"""
        model = _tiny_gpt2().eval()
        samples, old_logprobs, _ = self._sample_group(model)
        adv = torch.ones(2)
        old_small = old_logprobs - 5.0                 # ratio 爆炸（正优势 → 上界生效）
        loss_sym = compute_clipped_policy_loss(
            model, samples, old_small, adv, prompt_len=5, clip_eps=0.2)
        loss_high = compute_clipped_policy_loss(
            model, samples, old_small, adv, prompt_len=5, clip_eps=0.2,
            clip_eps_high=0.5)
        self.assertAlmostEqual(float(loss_sym), -1.2, places=3)
        self.assertAlmostEqual(float(loss_high), -1.5, places=3)

        neg_adv = -torch.ones(2)
        old_big = old_logprobs + 5.0                   # ratio≈0（负优势 → 下界生效）
        loss_neg_sym = compute_clipped_policy_loss(
            model, samples, old_big, neg_adv, prompt_len=5, clip_eps=0.2)
        loss_neg_high = compute_clipped_policy_loss(
            model, samples, old_big, neg_adv, prompt_len=5, clip_eps=0.2,
            clip_eps_high=0.5)
        self.assertAlmostEqual(float(loss_neg_sym), 0.8, places=3)
        self.assertAlmostEqual(float(loss_neg_high), float(loss_neg_sym), places=5)

    def test_k1_vs_k3_estimator(self):
        """k1=0.5·Δ² 与 k3=exp(Δ)-Δ-1 的 masked mean 手算比对（不同 ref 模型保证 Δ≠0）。"""
        model = _tiny_gpt2().eval()                    # seed 3
        ref = _tiny_gpt2(seed=4).eval()
        samples, old_logprobs, _ = self._sample_group(model)
        zero_adv = torch.zeros(2)                      # 策略项为 0，loss 即 KL 项
        loss_k3 = compute_clipped_policy_loss(
            model, samples, old_logprobs, zero_adv, prompt_len=5,
            kl_beta=1.0, ref_model=ref)
        loss_k1 = compute_clipped_policy_loss(
            model, samples, old_logprobs, zero_adv, prompt_len=5,
            kl_beta=1.0, ref_model=ref, kl_estimator="k1")
        log_ratio = token_logprobs(model, samples, start_pos=5) \
            - token_logprobs(ref, samples, start_pos=5)
        self.assertAlmostEqual(float(loss_k3), float((log_ratio.exp() - log_ratio - 1).mean()), places=5)
        self.assertAlmostEqual(float(loss_k1), float((0.5 * log_ratio.square()).mean()), places=5)
        self.assertGreater(abs(float(loss_k1) - float(loss_k3)), 1e-3)

    def test_seq_sum_mean_differs_from_token_mean(self):
        """变长序列（有效长 [4,2]，adv [1,-1]）下两种归一不同：
        token_mean=-(4-2)/6，seq_sum_mean=-(4-2)/2（Dr.GRPO 长度偏差修正）。"""
        model = _tiny_gpt2().eval()
        samples, old_logprobs, _ = self._sample_group(model)
        comp_lens = torch.tensor([4, 2])
        adv = torch.tensor([1.0, -1.0])
        # old_logprobs 来自同一模型 → ratio=1 → token_obj = adv
        loss_tm = compute_clipped_policy_loss(
            model, samples, old_logprobs, adv, prompt_len=5, clip_eps=0.2,
            completion_lengths=comp_lens, loss_norm="token_mean")
        loss_ss = compute_clipped_policy_loss(
            model, samples, old_logprobs, adv, prompt_len=5, clip_eps=0.2,
            completion_lengths=comp_lens, loss_norm="seq_sum_mean")
        self.assertAlmostEqual(float(loss_tm), -(4 * 1.0 + 2 * -1.0) / 6.0, places=4)
        self.assertAlmostEqual(float(loss_ss), -(4 * 1.0 + 2 * -1.0) / 2.0, places=4)

    def test_return_diagnostics_clip_frac(self):
        model = _tiny_gpt2().eval()
        samples, old_logprobs, comp_lens = self._sample_group(model)
        adv = torch.ones(2)
        # ratio 全部越上界且优势为正 → clip 分支全程生效
        _, diag = compute_clipped_policy_loss(
            model, samples, old_logprobs - 5.0, adv, prompt_len=5, clip_eps=0.2,
            completion_lengths=comp_lens, return_diagnostics=True)
        self.assertIsInstance(diag, dict)
        self.assertAlmostEqual(diag["clip_frac"], 1.0, places=4)
        # 自然 ratio≈1 → 不触发 clip
        _, diag2 = compute_clipped_policy_loss(
            model, samples, old_logprobs, adv, prompt_len=5, clip_eps=0.2,
            completion_lengths=comp_lens, return_diagnostics=True)
        self.assertLess(diag2["clip_frac"], 0.05)
        self.assertAlmostEqual(diag2["ratio_mean"], 1.0, places=3)
        self.assertAlmostEqual(diag2["kl"], 0.0, places=6)


class TestRewards(unittest.TestCase):
    def test_correct_answer_reward(self):
        reward_fn = correct_answer_reward()
        self.assertEqual(reward_fn(r"The answer is \boxed{42}", "42"), 1.0)
        self.assertEqual(reward_fn(r"\boxed{41}", "42"), 0.0)
        # fallback=None：无 \boxed{} 时即使文本含正确数字也不给分
        self.assertEqual(reward_fn("the answer is 42", "42"), 0.0)

    def test_correct_answer_reward_custom_fns(self):
        reward_fn = correct_answer_reward(
            extract_fn=lambda text, fallback=None: text.strip(),
            grade_fn=lambda pred, expected: pred == expected,
        )
        self.assertEqual(reward_fn("42", "42"), 1.0)
        self.assertEqual(reward_fn("42", "43"), 0.0)

    def test_format_reward(self):
        self.assertEqual(format_reward("<think>hmm</think> \\boxed{1}"), 1.0)
        self.assertEqual(format_reward("\\boxed{1}"), 0.5)
        self.assertEqual(format_reward("<think>hmm</think> answer"), 0.5)
        # 未闭合的 <think> 不算满足
        self.assertEqual(format_reward("<think>only"), 0.0)
        self.assertEqual(format_reward("no format at all"), 0.0)
        self.assertEqual(format_reward("\\boxed{1}", require_think=False), 1.0)
        self.assertEqual(format_reward("anything", require_think=False, require_boxed=False), 1.0)


class TestTracking(unittest.TestCase):
    def test_compute_clip_fraction(self):
        ratio = torch.tensor([[1.5, 0.5, 1.0]])
        adv_pos = torch.tensor([1.0])
        # 1.5 > 1.2 且 A>0 → 生效；0.5 < 0.8 但 A>0 → 不生效
        self.assertAlmostEqual(compute_clip_fraction(ratio, adv_pos, 0.2), 1 / 3, places=5)
        # clip_eps_high=1.0 → 上界 2.0，1.5 不再越界
        self.assertAlmostEqual(compute_clip_fraction(ratio, adv_pos, 0.2, clip_eps_high=1.0), 0.0, places=5)
        # 负优势：0.5 < 0.8 → 生效；1.5 > 1.2 但 A<0 → 不生效
        adv_neg = torch.tensor([-1.0])
        self.assertAlmostEqual(compute_clip_fraction(ratio, adv_neg, 0.2), 1 / 3, places=5)

    def test_metrics_tracker_csv_roundtrip(self):
        tracker = MetricsTracker()
        tracker.append(step=1, loss=0.5, reward_mean=0.25)
        tracker.append(step=2, loss=0.4, reward_mean=0.5)
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "metrics.csv"
            tracker.to_csv(csv_path)
            with csv_path.open(newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 2)
        self.assertEqual([r["step"] for r in rows], ["1", "2"])
        self.assertAlmostEqual(float(rows[1]["loss"]), 0.4)
        summary = tracker.summary()
        self.assertAlmostEqual(summary["reward_mean"]["mean"], 0.375)
        self.assertEqual(summary["step"]["last"], 2)

    def test_plot_metrics_missing_matplotlib(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "metrics.csv"
            tracker = MetricsTracker()
            tracker.append(step=1, loss=1.0)
            tracker.append(step=2, loss=0.5)
            tracker.to_csv(csv_path)
            out_png = Path(tmp) / "plot.png"
            try:
                import matplotlib  # noqa: F401
            except ImportError:
                with self.assertRaises(ImportError):
                    plot_metrics(csv_path, out_png)
            else:
                self.assertTrue(plot_metrics(csv_path, out_png).exists())


class TestBatched(unittest.TestCase):
    def test_greedy_matches_serial(self):
        """greedy（temperature=0）下批量版与串行版输出逐位一致。"""
        torch.manual_seed(7)
        model = _tiny_gpt2().eval()
        x = torch.randint(0, 32, (1, 5))
        s1, o1, l1 = generate_group_samples(
            model, x, group_size=4, max_new_tokens=6, context_length=48,
            temperature=0.0)
        s2, o2, l2 = generate_group_samples_batched(
            model, x, group_size=4, max_new_tokens=6, context_length=48,
            temperature=0.0)
        self.assertTrue(torch.equal(s1, s2))
        self.assertTrue(torch.equal(o1, o2))
        self.assertTrue(torch.equal(l1, l2))

    def test_greedy_eos_matches_serial(self):
        model = _tiny_gpt2().eval()
        x = torch.randint(0, 32, (1, 3))
        with torch.no_grad():
            first_token = int(torch.argmax(model(x)[:, -1, :], dim=-1).item())
        kwargs = dict(group_size=3, max_new_tokens=8, context_length=48,
                      temperature=0.0, eos_token_id=first_token)
        s1, o1, l1 = generate_group_samples(model, x, **kwargs)
        s2, o2, l2 = generate_group_samples_batched(model, x, **kwargs)
        self.assertTrue(torch.equal(s1, s2))
        self.assertTrue(torch.equal(o1, o2))
        self.assertTrue(torch.equal(l1, l2))
        self.assertTrue(bool((l1 == 1).all()))

    def test_pad_id_fills_after_eos(self):
        model = _tiny_gpt2().eval()
        x = torch.randint(0, 32, (1, 3))
        with torch.no_grad():
            first_token = int(torch.argmax(model(x)[:, -1, :], dim=-1).item())
        samples, old_logprobs, comp_lens = generate_group_samples_batched(
            model, x, group_size=3, max_new_tokens=5, context_length=48,
            temperature=0.0, eos_token_id=first_token, pad_id=0)
        self.assertTrue(bool((comp_lens == 1).all()))
        self.assertTrue(bool((samples[:, 4:] == 0).all()))   # EOS 后全部 pad

    def test_grpo_update_step_batched_runs(self):
        torch.manual_seed(11)
        model = _tiny_gpt2()
        x = torch.randint(0, 32, (1, 5))
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)
        reward_fn = lambda ids_row: 1.0 if int(ids_row[-1].item()) > 16 else 0.0
        res = grpo_update_step_batched(
            model, optimizer, x, reward_fn, group_size=6, max_new_tokens=4,
            context_length=48, temperature=1.0,
        )
        self.assertTrue(torch.isfinite(torch.tensor(res["loss"])))
        self.assertEqual(len(res["rewards"]), 6)
        self.assertGreaterEqual(res["mean_reward"], 0.0)


class TestGRPOVariantLosses(unittest.TestCase):
    """grpo_variants 三个变体：可运行 + 关键数学方向性。"""

    def test_olmo3_clip_higher_widens_bound(self):
        from post_training.rl.grpo_variants import compute_olmo3_style_loss

        model = _tiny_gpt2().eval()
        x = torch.randint(0, 32, (1, 5))
        samples, old_logprobs, comp_lens = generate_group_samples(
            model, x, group_size=2, max_new_tokens=4, context_length=48,
            temperature=1.0)
        adv = torch.ones(2)
        old_small = old_logprobs - 5.0
        # ratio 巨大 → 截断 IS 到 1+clip_eps_high；正优势 → loss=-(1+clip_eps_high)
        loss = compute_olmo3_style_loss(
            model, samples, old_small, adv, prompt_len=5,
            clip_eps=0.2, clip_eps_high=0.5, completion_lengths=comp_lens)
        self.assertAlmostEqual(float(loss), -1.5, places=3)

    def test_deepseek_v32_off_policy_masking(self):
        from post_training.rl.grpo_variants import compute_deepseek_v32_style_loss
        import math

        model = _tiny_gpt2().eval()
        x = torch.randint(0, 32, (1, 5))
        samples, old_logprobs, comp_lens = generate_group_samples(
            model, x, group_size=2, max_new_tokens=4, context_length=48,
            temperature=1.0)
        adv = torch.tensor([1.0, -1.0])
        old_big = old_logprobs + 1.0                 # seq_kl=(old-new)/len=1 > 0.1
        loss_masked = compute_deepseek_v32_style_loss(
            model, samples, old_big, adv, prompt_len=5,
            kl_beta=0.0, off_policy_delta=0.1, completion_lengths=comp_lens)
        loss_unmasked = compute_deepseek_v32_style_loss(
            model, samples, old_big, adv, prompt_len=5,
            kl_beta=0.0, off_policy_delta=1e9, completion_lengths=comp_lens)
        # 负优势样本被 off-policy 掩码置 0 → 序列级 ratio=e^-4 只乘正优势
        self.assertAlmostEqual(float(loss_masked), -math.exp(-4.0) / 2.0, places=4)
        self.assertAlmostEqual(float(loss_unmasked), 0.0, places=4)
        self.assertLess(float(loss_masked), float(loss_unmasked))

    def test_gdpo_advantages_and_step(self):
        from post_training.rl.grpo_variants import (
            compute_gdpo_advantages,
            gdpo_update_step,
        )

        adv = compute_gdpo_advantages(
            correctness_rewards=[1.0, 0.0, 1.0, 0.0],
            format_rewards=[1.0, 1.0, 0.0, 0.0],
            format_coeff=0.1,
        )
        self.assertAlmostEqual(float(adv.mean()), 0.0, places=5)

        torch.manual_seed(5)
        model = _tiny_gpt2()
        x = torch.randint(0, 32, (1, 4))
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        res = gdpo_update_step(
            model, optimizer, x,
            correctness_reward_fn=lambda ids_row: 1.0 if int(ids_row[-1].item()) > 16 else 0.0,
            format_reward_fn=lambda ids_row: 1.0 if int(ids_row[0].item()) % 2 == 0 else 0.0,
            group_size=4, max_new_tokens=3, context_length=48, temperature=1.0,
            kl_beta=0.0,
        )
        self.assertTrue(torch.isfinite(torch.tensor(res["loss"])))

    def test_olmo3_zero_advantage_skip(self):
        from post_training.rl.grpo_variants import olmo3_update_step

        torch.manual_seed(5)
        model = _tiny_gpt2()
        x = torch.randint(0, 32, (1, 4))
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        res = olmo3_update_step(
            model, optimizer, x,
            reward_fn=lambda ids_row: 1.0,   # 恒定 reward → 优势全 0
            group_size=4, max_new_tokens=3, context_length=48, temperature=1.0,
        )
        self.assertTrue(res["zero_advantage"])
        self.assertEqual(res["loss"], 0.0)


if __name__ == "__main__":
    unittest.main()
