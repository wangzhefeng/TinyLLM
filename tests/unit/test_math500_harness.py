"""MATH-500 评测 harness 测试（完全离线：注入替身生成函数，不加载真实模型）。"""
import json
import tempfile
import unittest
from pathlib import Path

import torch

from inference.evaluation.benchmarks.math_verifiers import render_prompt
from inference.evaluation.harness.math500 import (
    evaluate_math500,
    evaluate_math500_batched,
    heuristic_score,
    save_report,
)
from inference.strategies.advance_text_generation import (
    generate_with_self_refinement,
)
from inference.strategies.voting_and_verification import (
    best_of_n,
    extract_boxed_answer,
    self_consistency,
    weighted_self_consistency,
)


class _MockCharTokenizer:
    """字符级 mock 分词器：ord(ch) % vocab_size，encode/decode 互逆。"""

    def __init__(self, vocab_size=256):
        self.vocab_size = vocab_size
        self.pad_token_id = 0

    def encode(self, text):
        return [ord(ch) % self.vocab_size for ch in text]

    def decode(self, ids):
        return "".join(chr(i) for i in ids)


class _CountingGen:
    """按预设答案序列生成 \\boxed 文本，并计数调用与收到的 prompt ids。"""

    def __init__(self, answers):
        self.answers = answers
        self.calls = 0
        self.prompts = []

    def __call__(self, token_ids):
        self.calls += 1
        self.prompts.append(token_ids[0].tolist())
        answer = self.answers[(self.calls - 1) % len(self.answers)]
        return f"step by step ... therefore \\boxed{{{answer}}}"


PROBLEMS = [
    {"problem": "What is 2 + 2?", "answer": "4"},
    {"problem": "What is 3 + 3?", "answer": "6"},
    {"problem": "What is 5 + 5?", "answer": "10"},
    {"problem": "What is 7 + 7?", "answer": "14"},
]


class TestRenderPromptCot(unittest.TestCase):
    def test_default_template_unchanged(self):
        self.assertEqual(
            render_prompt("q"),
            "You are a helpful math assistant.\n"
            "Answer the question and write the final result on a new line as:\n"
            "\\boxed{ANSWER}\n\n"
            "Question:\nq\n\nAnswer:",
        )

    def test_cot_appends_instruction(self):
        base = render_prompt("q")
        cot = render_prompt("q", cot=True)
        self.assertNotIn("step by step", base)
        self.assertTrue(cot.startswith(base))
        self.assertEqual(cot, base + " \n\nExplain step by step.")


class TestEvaluateMath500(unittest.TestCase):
    def test_accuracy_and_report_structure(self):
        tokenizer = _MockCharTokenizer()
        gen = _CountingGen(["4", "5", "10", "15"])      # 2 对 2 错
        report = evaluate_math500(None, tokenizer, PROBLEMS, generate_fn=gen)

        self.assertEqual(set(report.keys()), {"accuracy", "num_problems", "results"})
        self.assertEqual(report["num_problems"], 4)
        self.assertAlmostEqual(report["accuracy"], 0.5)
        self.assertEqual(gen.calls, 4)
        self.assertEqual(len(report["results"]), 4)

        for result, problem in zip(report["results"], PROBLEMS):
            self.assertEqual(
                set(result.keys()),
                {"problem", "expected", "pred", "raw", "correct"},
            )
            self.assertEqual(result["problem"], problem["problem"])
            self.assertEqual(result["expected"], problem["answer"])
            self.assertIsInstance(result["correct"], bool)
        # pred 来自 \\boxed 抽取；raw 保留完整生成文本
        self.assertEqual(report["results"][0]["pred"], "4")
        self.assertEqual(report["results"][1]["pred"], "5")
        self.assertIn("\\boxed{5}", report["results"][1]["raw"])

    def test_generate_fn_receives_encoded_prompt_ids(self):
        tokenizer = _MockCharTokenizer()
        gen = _CountingGen(["4"])
        evaluate_math500(None, tokenizer, PROBLEMS[:1], generate_fn=gen)
        prompt_ids = gen.prompts[0]
        self.assertTrue(prompt_ids)
        self.assertTrue(all(isinstance(i, int) for i in prompt_ids))
        # 编码可由 mock 分词器解码回原文
        self.assertIn("What is 2 + 2?", tokenizer.decode(prompt_ids))

    def test_cot_flag_changes_encoded_prompt(self):
        tokenizer = _MockCharTokenizer()
        gen_plain = _CountingGen(["4"])
        gen_cot = _CountingGen(["4"])
        evaluate_math500(None, tokenizer, PROBLEMS[:1], generate_fn=gen_plain)
        evaluate_math500(None, tokenizer, PROBLEMS[:1], generate_fn=gen_cot, cot=True)
        self.assertNotEqual(gen_plain.prompts[0], gen_cot.prompts[0])

    def test_empty_problems(self):
        tokenizer = _MockCharTokenizer()
        report = evaluate_math500(None, tokenizer, [], generate_fn=_CountingGen([]))
        self.assertEqual(report["num_problems"], 0)
        self.assertEqual(report["accuracy"], 0.0)
        self.assertEqual(report["results"], [])

    def test_out_path_saves_report(self):
        tokenizer = _MockCharTokenizer()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nested" / "math500.json"
            report = evaluate_math500(
                None, tokenizer, PROBLEMS[:2],
                generate_fn=_CountingGen(["4", "6"]), out_path=path,
            )
            self.assertTrue(path.exists())
            loaded = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(loaded, report)


class TestSaveReport(unittest.TestCase):
    def test_roundtrip_creates_parent_dirs(self):
        tokenizer = _MockCharTokenizer()
        report = evaluate_math500(
            None, tokenizer, PROBLEMS[:2], generate_fn=_CountingGen(["4", "6"]),
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "a" / "b" / "report.json"
            returned = save_report(report, path)
            self.assertEqual(returned, path)
            loaded = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(loaded, report)


class TestEvaluateMath500Batched(unittest.TestCase):
    def _fake_batched(self, canned, batches):
        state = {"sent": 0}

        def fn(token_ids, attn_mask, pad_id):
            rows = token_ids.shape[0]
            texts = canned[state["sent"]:state["sent"] + rows]
            state["sent"] += rows
            # 右填充：每行首个 token 必须未被 mask
            self.assertTrue(bool(attn_mask[:, 0].all()))
            batches.append((rows, pad_id))
            return texts

        return fn

    def test_injected_batched_fn_counts(self):
        tokenizer = _MockCharTokenizer()
        canned = [r"\boxed{4}", r"\boxed{6}", r"\boxed{11}", r"\boxed{14}"]
        batches = []
        report = evaluate_math500_batched(
            None, tokenizer, PROBLEMS, batch_size=2,
            batched_generate_fn=self._fake_batched(canned, batches),
        )
        self.assertEqual(report["num_problems"], 4)
        self.assertAlmostEqual(report["accuracy"], 0.75)   # 11 vs 10 判错
        self.assertEqual(len(batches), 2)
        self.assertTrue(all(rows == 2 for rows, _ in batches))
        self.assertTrue(all(pad_id == 0 for _, pad_id in batches))
        # 行对齐：预置文本与题目一一对应
        self.assertEqual(report["results"][0]["pred"], "4")
        self.assertEqual(report["results"][1]["pred"], "6")
        self.assertEqual(report["results"][2]["pred"], "11")
        self.assertTrue(report["results"][0]["correct"])
        self.assertFalse(report["results"][2]["correct"])

    def test_tail_batch_smaller_than_batch_size(self):
        tokenizer = _MockCharTokenizer()
        canned = [r"\boxed{4}"] * 5
        batches = []
        report = evaluate_math500_batched(
            None, tokenizer, PROBLEMS + [PROBLEMS[0]], batch_size=2,
            batched_generate_fn=self._fake_batched(canned, batches),
        )
        self.assertEqual(report["num_problems"], 5)
        self.assertEqual([rows for rows, _ in batches], [2, 2, 1])

    def test_row_count_mismatch_raises(self):
        tokenizer = _MockCharTokenizer()

        def bad_fn(token_ids, attn_mask, pad_id):
            return [r"\boxed{4}"] * (token_ids.shape[0] + 1)

        with self.assertRaises(ValueError):
            evaluate_math500_batched(
                None, tokenizer, PROBLEMS[:2], batch_size=2,
                batched_generate_fn=bad_fn,
            )


class TestStrategyGenerateFnInjection(unittest.TestCase):
    def test_self_consistency_uses_generate_fn(self):
        texts = iter([r"\boxed{4}", r"\boxed{4}", r"\boxed{6}"])
        majority, answers = self_consistency(
            None, None, num_samples=3,
            extract_answer=extract_boxed_answer,
            generate_fn=lambda ids: next(texts), return_all=True,
        )
        self.assertEqual(majority, "4")
        self.assertEqual(len(answers), 3)

    def test_best_of_n_uses_generate_fn(self):
        texts = iter([r"\boxed{1}", r"\boxed{9}", r"\boxed{5}"])
        best_text, best_reward, scored = best_of_n(
            None, None, lambda t: float(extract_boxed_answer(t)),
            num_samples=3, generate_fn=lambda ids: next(texts), return_all=True,
        )
        self.assertEqual(best_reward, 9.0)
        self.assertEqual(extract_boxed_answer(best_text), "9")
        self.assertEqual(len(scored), 3)

    def test_self_refinement_uses_generate_fn(self):
        x = torch.tensor([[1, 2, 3]])
        drafts = iter(["plain text", r"done \boxed{8}"])
        prompts_seen = []

        def gen(token_ids):
            prompts_seen.append(token_ids)
            return next(drafts)

        def refine(_draft_ids, text):
            if "\\boxed" in text:
                return True, None, None
            return False, x, None          # 未给出 boxed：原样重问

        final_ids, rounds, log = generate_with_self_refinement(
            None, x, refine, max_rounds=3, generate_fn=gen,
        )
        self.assertEqual(rounds, 2)
        self.assertIsNone(final_ids)       # 注入文本生成时不回传 token ids
        self.assertTrue(log[-1][1])
        self.assertIs(prompts_seen[1], x)  # 第二轮从原 prompt 重问


class TestWeightedSelfConsistency(unittest.TestCase):
    @staticmethod
    def _gen_from_texts(texts):
        state = {"i": 0}

        def gen(token_ids):
            text = texts[state["i"]]
            state["i"] += 1
            return text

        return gen

    def test_scorer_bias_beats_majority(self):
        texts = [r"\boxed{7}", r"\boxed{7}", r"\boxed{9}"]
        scorer = lambda t: 10.0 if "\\boxed{9}" in t else 1.0   # noqa: E731
        winner, samples = weighted_self_consistency(
            None, None, scorer, num_samples=3,
            extract_answer=extract_boxed_answer,
            generate_fn=self._gen_from_texts(texts), return_all=True,
        )
        # 多数票本应是 "7"，加权后被评分器偏向的 "9" 胜出
        self.assertEqual(winner, "9")
        self.assertEqual(len(samples), 3)
        self.assertEqual([s for _, _, s in samples], [1.0, 1.0, 10.0])

    def test_all_zero_scores_fall_back_to_majority(self):
        texts = [r"\boxed{7}", r"\boxed{7}", r"\boxed{9}"]
        winner = weighted_self_consistency(
            None, None, lambda t: 0.0, num_samples=3,
            extract_answer=extract_boxed_answer,
            generate_fn=self._gen_from_texts(texts),
        )
        self.assertEqual(winner, "7")

    def test_positive_tie_scores_pick_earliest(self):
        texts = [r"\boxed{9}", r"\boxed{7}"]
        winner = weighted_self_consistency(
            None, None, lambda t: 1.0, num_samples=2,
            extract_answer=extract_boxed_answer,
            generate_fn=self._gen_from_texts(texts),
        )
        self.assertEqual(winner, "9")      # 总权重并列时取首次出现最早


class TestHeuristicScore(unittest.TestCase):
    def test_boxed_beats_number_fallback(self):
        boxed = heuristic_score(r"so the answer is \boxed{42}")
        plain = heuristic_score("so the answer is 42")
        self.assertGreater(boxed, plain)

    def test_always_positive(self):
        self.assertGreater(heuristic_score("no numbers at all"), 0.0)


if __name__ == "__main__":
    unittest.main()
