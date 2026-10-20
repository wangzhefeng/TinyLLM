"""推断时扩展脚手架测试：self-consistency / Best-of-N / self-refinement（Phase 2 推断域）。"""
import unittest
from types import SimpleNamespace

import torch

from post_training.reasoning.inference.advance_text_generation import (
    generate_with_self_refinement,
    refine_until_stable,
)
from post_training.reasoning.inference.voting_and_verification import (
    best_of_n,
    extract_boxed_answer,
    self_consistency,
)
from models.gpt2.gpt2_124M import Model


def _tiny_gpt2(vocab=32):
    torch.manual_seed(7)
    cfg = SimpleNamespace(
        vocab_size=vocab, context_length=48, embed_dim=32, d_ff=64,
        n_heads=4, n_layers=2, dropout=0.0, qkv_bias=True,
        dtype=torch.float32, is_train=False,
    )
    return Model(cfg).eval()


def _last_number(text: str) -> int:
    """从解码文本取最后一个数字（模拟规范化答案抽取）。"""
    return int(text.split()[-1])


class TestSelfConsistency(unittest.TestCase):
    def test_majority_vote(self):
        model = _tiny_gpt2()
        x = torch.randint(0, 32, (1, 5))
        majority, answers = self_consistency(
            model, x, num_samples=5, extract_answer=_last_number,
            max_new_tokens=3, context_length=48, temperature=1.0, return_all=True,
        )
        self.assertEqual(len(answers), 5)
        # 多数答案确实出现次数最多
        from collections import Counter
        counts = Counter(a for _, a in answers)
        self.assertEqual(counts[majority], max(counts.values()))


class TestBestOfN(unittest.TestCase):
    def test_selects_max_reward(self):
        model = _tiny_gpt2()
        x = torch.randint(0, 32, (1, 5))
        # reward = 最后 token 的 id（有客观标准），BoN 应选出采样中最大的
        best_text, best_reward, scored = best_of_n(
            model, x, reward_fn=lambda t: float(t.split()[-1]),
            num_samples=6, max_new_tokens=3, context_length=48,
            temperature=1.0, return_all=True,
        )
        self.assertEqual(best_reward, max(r for _, r in scored))
        self.assertEqual(len(scored), 6)


class TestSelfRefinement(unittest.TestCase):
    def test_accepts_when_validator_passes(self):
        model = _tiny_gpt2()
        x = torch.randint(0, 32, (1, 5))

        def refine_fn(draft_ids, draft_text):
            last = draft_ids[0, -1].item()
            if last % 2 == 0:                       # 验证器：偶数通过
                return True, None, None
            # 反馈 prompt：原 prompt + 一条修正 token，诱导重生成
            feedback = torch.cat([x, torch.tensor([[1]])], dim=1)
            return False, feedback, draft_text

        final, rounds, log = generate_with_self_refinement(
            model, x, refine_fn, max_new_tokens=2, max_rounds=5,
            context_length=48, temperature=1.0,
        )
        self.assertLessEqual(rounds, 5)
        self.assertTrue(log[-1][1])                 # 最后一轮必须通过
        self.assertEqual(final[0, -1].item() % 2, 0)

    def test_stops_at_max_rounds(self):
        model = _tiny_gpt2()
        x = torch.randint(0, 32, (1, 5))
        never_accept = lambda ids, text: (False, x, text)   # noqa: E731
        _, rounds, log = generate_with_self_refinement(
            model, x, never_accept, max_new_tokens=2, max_rounds=3,
            context_length=48, temperature=1.0,
        )
        self.assertEqual(rounds, 3)
        self.assertEqual(len(log), 3)


class TestRefineUntilStable(unittest.TestCase):
    def test_early_stop_on_stable_answer(self):
        model = _tiny_gpt2()
        x = torch.randint(0, 32, (1, 5))
        # greedy（temperature=0）：每轮答案必然相同 -> 第 2 轮即稳定
        _, rounds, answers = refine_until_stable(
            model, x, max_new_tokens=2, max_rounds=6, stable_threshold=2,
            context_length=48, temperature=0.0,
        )
        self.assertEqual(rounds, 2)
        self.assertEqual(len(answers), 2)
        self.assertEqual(answers[0], answers[1])


class TestBoxedExtraction(unittest.TestCase):
    def test_boxed(self):
        self.assertEqual(extract_boxed_answer(r"so \boxed{42} done"), "42")
        self.assertEqual(extract_boxed_answer(r"\boxed{-3.5}"), "-3.5")

    def test_fallback_number(self):
        self.assertEqual(extract_boxed_answer("3 + 4 = 7"), "7")
        self.assertIsNone(extract_boxed_answer("no numbers"))


if __name__ == "__main__":
    unittest.main()
