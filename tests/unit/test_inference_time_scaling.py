"""推断时扩展脚手架测试：self-consistency / Best-of-N / self-refinement（Phase 2 推断域）。
Phase 4 扩展：advanced_parser（混合 LaTeX 解析器）与 MMLU teacher-forcing 评测。"""
import unittest
from types import SimpleNamespace

import torch

from inference.strategies.advance_text_generation import (
    generate_with_self_refinement,
    refine_until_stable,
)
from inference.strategies.voting_and_verification import (
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

    def test_boxed_nested_braces(self):
        r"""嵌套大括号（修复目标）：\boxed{\dfrac{14}{3}} 不再回退取最后一个数字。"""
        self.assertEqual(
            extract_boxed_answer(r"x = \boxed{\dfrac{14}{3}}"),
            r"\dfrac{14}{3}",
        )
        self.assertEqual(
            extract_boxed_answer(r"\boxed{\frac{1}{2}} + \boxed{\frac{3}{4}}"),
            r"\frac{3}{4}",                          # 取最后一个 boxed
        )

    def test_fallback_number(self):
        self.assertEqual(extract_boxed_answer("3 + 4 = 7"), "7")
        self.assertIsNone(extract_boxed_answer("no numbers"))


    def test_boxed_truncated_tail_falls_back_to_previous_complete(self):
        """max_tokens 截断在未闭合 \\boxed 上时，回退取上一个完整 boxed。"""
        from evaluation.verifiers.math import (
            extract_final_candidate,
            get_last_boxed,
        )
        text = r"Answer: $\boxed{(3, \frac{\pi}{3})}$ repeat $\boxed{(3, \frac{\pi}{3})}$ $\boxed{(3"
        self.assertEqual(get_last_boxed(text), r"(3, \frac{\pi}{3})")
        self.assertEqual(extract_final_candidate(text), r"(3, \frac{\pi}{3})")

    def test_boxed_all_truncated_returns_none(self):
        from evaluation.verifiers.math import (
            get_last_boxed,
        )
        self.assertIsNone(get_last_boxed(r"$\boxed{1"))
        self.assertIsNone(get_last_boxed("no boxed at all"))


class TestMajorityVoteTieRule(unittest.TestCase):
    """平票规则：并列最高票时取首次出现最早的答案（不再静默依赖排序行为）。"""

    def test_tie_picks_earliest_occurrence(self):
        from inference.strategies.voting_and_verification import (
            _majority_vote,
        )
        self.assertEqual(_majority_vote(["b", "a", "b", "a"]), "b")
        self.assertEqual(_majority_vote(["a", "b", "b", "a"]), "a")

    def test_clear_majority(self):
        from inference.strategies.voting_and_verification import (
            _majority_vote,
        )
        self.assertEqual(_majority_vote(["x", "x", "y"]), "x")
        majority, counts = _majority_vote(["x", "x", "y"], return_counts=True)
        self.assertEqual(majority, "x")
        self.assertEqual(counts["y"], 1)

    def test_self_consistency_return_counts(self):
        """return_counts=True 时不改变原有返回形状的前提下附带票数。"""
        model = _tiny_gpt2()
        x = torch.randint(0, 32, (1, 5))
        majority, answers, counts = self_consistency(
            model, x, num_samples=4, extract_answer=_last_number,
            max_new_tokens=3, context_length=48, temperature=1.0,
            return_all=True, return_counts=True,
        )
        self.assertEqual(len(answers), 4)
        self.assertEqual(sum(counts.values()), 4)
        self.assertEqual(counts[majority], max(counts.values()))


class TestMathVerifierNormalize(unittest.TestCase):
    """normalize_text 补齐原书两段逻辑：Unicode 上标与多项选择标签剥离。"""

    def test_superscript_conversion(self):
        from evaluation.verifiers.math import (
            normalize_text,
        )
        self.assertEqual(normalize_text("2²"), "2**2")
        self.assertEqual(normalize_text("x³"), "x**3")
        self.assertEqual(normalize_text("10⁻¹"), "10**-1")

    def test_multiple_choice_label_strip(self):
        from evaluation.verifiers.math import (
            normalize_text,
        )
        self.assertEqual(normalize_text("B. 7"), "7")
        self.assertEqual(normalize_text("c: 2"), "2")

    def test_grade_answer_with_new_normalization(self):
        from evaluation.verifiers.math import (
            grade_answer,
        )
        self.assertTrue(grade_answer("B. 7", "7"))
        self.assertTrue(grade_answer("2²", "4"))
        self.assertTrue(grade_answer("x²", "x**2"))
        # 回归：原有等价判断不受影响
        self.assertTrue(grade_answer(r"\frac{14}{3}", "14/3"))
        self.assertFalse(grade_answer("1/2", "1/3"))


class _MockCharTokenizer:
    """字符级 mock 分词器：ord(ch) % vocab_size，encode/decode 互逆。"""

    def __init__(self, vocab_size=32):
        self.vocab_size = vocab_size

    def encode(self, text):
        return [ord(ch) % self.vocab_size for ch in text]

    def decode(self, ids):
        return "".join(chr(i) for i in ids)


class _RiggedLM(torch.nn.Module):
    """恒定 logits 的 mock 模型：只在 favored_token_id 上给高分。"""

    def __init__(self, vocab_size=32, favored_token_id=2):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.zeros(vocab_size))
        with torch.no_grad():
            self.bias[favored_token_id] = 10.0

    def forward(self, x):
        b, t = x.shape
        return self.bias.view(1, 1, -1).expand(b, t, -1)


class TestMultipleChoiceLogprob(unittest.TestCase):
    """多项选择 Method 2（first-token logprob，书 chF/02 的 2_logprob.py）。"""

    def _example(self, answer):
        return {
            "question": "1 + 1 = ?",
            "choices": ["1", "2", "3", "4"],
            "answer": answer,
        }

    def test_predict_choice_logprob_favors_rigged_token(self):
        """chr('B')=66, 66%32=2：把 favored token 钉在 id=2 应稳定预测 B。"""
        from inference.harness.mmlu import (
            predict_choice_logprob,
        )
        model = _RiggedLM(vocab_size=32, favored_token_id=2)
        tokenizer = _MockCharTokenizer(vocab_size=32)
        prompt = "Q\nA. 1\nB. 2\nC. 3\nD. 4\nAnswer: "
        prompt_ids = tokenizer.encode(prompt)
        prompt_fmt = torch.tensor(prompt_ids).unsqueeze(0)

        pred, scores = predict_choice_logprob(model, tokenizer, prompt_fmt)
        self.assertEqual(pred, "B")
        self.assertEqual(set(scores), set("ABCD"))
        self.assertTrue(all(v == v for v in scores.values()))    # 无 NaN
        self.assertGreater(scores["B"], scores["A"])

    def test_first_new_token_id_alignment(self):
        """continuation 产生的新 token 需经公共前缀对齐（BPE 合并差异处理）。"""
        from inference.harness.mmlu import (
            common_prefix_len,
            first_new_token_id,
        )
        tokenizer = _MockCharTokenizer(vocab_size=32)
        prompt = "Answer: "
        prompt_ids = tokenizer.encode(prompt)
        # 字符级 tokenizer 下 "A" 的新 token id 即 ord('A') % 32
        self.assertEqual(first_new_token_id(tokenizer, prompt, prompt_ids, "A"), 65 % 32)
        self.assertEqual(common_prefix_len([1, 2, 3], [1, 2, 9]), 2)

    def test_evaluate_mmlu_int_and_str_answer(self):
        """评测循环：answer 兼容 int（映射 ABCD）与 str；mock 模型可预测。"""
        from inference.harness.mmlu import (
            evaluate_mmlu,
            predict_choice_logprob,
        )
        model = _RiggedLM(vocab_size=32, favored_token_id=2)     # 恒预测 B
        tokenizer = _MockCharTokenizer(vocab_size=32)
        data = [self._example(1), self._example("B")]            # gold 均为 B

        results = evaluate_mmlu(
            model, tokenizer, data,
            predict_fn=lambda m, t, p: predict_choice_logprob(m, t, p)[0],
            verbose_every=0,
        )
        self.assertEqual(results["num_examples"], 2)
        self.assertEqual(results["num_correct"], 2)
        self.assertEqual(results["accuracy"], 1.0)

    def test_evaluate_mmlu_max_examples(self):
        from inference.harness.mmlu import (
            evaluate_mmlu,
        )
        model = _RiggedLM(vocab_size=32)
        tokenizer = _MockCharTokenizer(vocab_size=32)
        data = [self._example(i % 4) for i in range(10)]
        results = evaluate_mmlu(
            model, tokenizer, data,
            predict_fn=lambda m, t, p: "A",
            max_examples=3, verbose_every=0,
        )
        self.assertEqual(results["num_examples"], 3)

    def test_random_guess_baseline(self):
        from inference.harness.mmlu import (
            random_guess_baseline,
        )
        b = random_guess_baseline(num_questions=500, num_choices=4, seed=123)
        self.assertEqual(b["expected"], 0.25)
        self.assertTrue(0.0 <= b["empirical"] <= 1.0)
        # 大数定律：实测值应接近期望
        self.assertAlmostEqual(b["empirical"], 0.25, delta=0.08)
        # 固定种子可复现
        b2 = random_guess_baseline(num_questions=500, num_choices=4, seed=123)
        self.assertEqual(b["empirical"], b2["empirical"])

    def test_gold_letter(self):
        from inference.harness.mmlu import (
            gold_letter,
        )
        self.assertEqual(gold_letter(0), "A")
        self.assertEqual(gold_letter(2), "C")
        self.assertEqual(gold_letter("b"), "B")
        self.assertEqual(gold_letter("D"), "D")


class TestAdvancedParser(unittest.TestCase):
    """advanced_parser（bonus/parser.py 移植）离线用例。

    本环境未安装 antlr4-python3-runtime：parse_latex 调用失败返回 None，
    latex 命令类输入经 parse_and_verify_answer 回退 basic grade_answer；
    断言只覆盖两条路径下都确定成立的行为。
    """

    def test_numeric_equivalence(self):
        from evaluation.verifiers.parser import (
            parse_and_verify_answer,
        )
        self.assertTrue(parse_and_verify_answer("2/3", "4/6"))
        self.assertTrue(parse_and_verify_answer("0.5", "1/2"))
        self.assertTrue(parse_and_verify_answer("13/4", "3.25"))
        self.assertTrue(parse_and_verify_answer("2**3", "8"))
        self.assertTrue(parse_and_verify_answer("sqrt(2)", "2**(1/2)"))
        self.assertFalse(parse_and_verify_answer("1/2", "1/3"))
        self.assertFalse(parse_and_verify_answer("2", "3"))

    def test_text_answer_format(self):
        from evaluation.verifiers.parser import (
            normalize_text_hybrid,
            parse_and_verify_answer,
            sympy_parser_hybrid,
        )
        self.assertTrue(parse_and_verify_answer(r"\text{42}", "42"))
        self.assertTrue(parse_and_verify_answer(r"\mbox{seven}", "SEVEN"))
        self.assertEqual(normalize_text_hybrid(r"\text{42}"), "42")
        self.assertIsInstance(sympy_parser_hybrid(r"\text{42}"), str)
        self.assertIsNone(sympy_parser_hybrid(""))

    def test_latex_falls_back_to_basic_grader(self):
        """\\frac 等 latex 命令：无 antlr 时解析失败回退 basic 评分；
        有 antlr 时直接解析为 sympy 对象（ Rational(14, 3)）。"""
        import sympy
        from evaluation.verifiers.parser import (
            parse_and_verify_answer,
            sympy_parser_hybrid,
        )
        try:
            from sympy.parsing.latex import parse_latex  # noqa: F401

            has_antlr = True
        except ImportError:
            has_antlr = False
        if has_antlr:
            self.assertEqual(
                sympy_parser_hybrid(r"\frac{14}{3}"), sympy.Rational(14, 3),
            )
        else:
            self.assertIsNone(sympy_parser_hybrid(r"\frac{14}{3}"))
        self.assertTrue(parse_and_verify_answer(r"\frac{14}{3}", "14/3"))
        self.assertTrue(parse_and_verify_answer(r"\frac{3}{4}", "0.75"))
        self.assertTrue(parse_and_verify_answer(r"\sqrt{8}", "sqrt(8)"))

    def test_tuple_and_interval_like_formats(self):
        """元组/区间风格输入：括号容器经 sympify 结构比较；开放性差异交给 basic。"""
        from evaluation.verifiers.parser import (
            normalize_text_hybrid,
            parse_and_verify_answer,
        )
        self.assertTrue(parse_and_verify_answer("(0, 1)", "(0,1)"))
        self.assertFalse(parse_and_verify_answer("(0, 1)", "(0,2)"))
        self.assertFalse(parse_and_verify_answer("(1, 2)", "(2, 1)"))
        self.assertEqual(normalize_text_hybrid("(0, 1)"), "(0, 1)")
        # 区间开放性（开/闭端点）语义需要 antlr4；未安装时两条路径都归 basic
        # 分部比较（忽略开闭），此处仅断言不抛异常且可给出布尔结果
        self.assertIsInstance(
            parse_and_verify_answer("[0, 1)", "(0,1]"), bool,
        )

    def test_multi_answer_comma_formats(self):
        """多答案逗号列表可比较；\\{..\\} 集合在有 antlr 时解析为 set 并与
        逗号列表按多重集等价；缺失时两侧解析失败确定判 False（降级行为）。"""
        from evaluation.verifiers.parser import (
            parse_and_verify_answer,
        )
        try:
            from sympy.parsing.latex import parse_latex  # noqa: F401

            has_antlr = True
        except ImportError:
            has_antlr = False
        self.assertTrue(parse_and_verify_answer("1, 2", "1,2"))
        self.assertTrue(parse_and_verify_answer("1, 2", "1, 2"))
        self.assertFalse(parse_and_verify_answer("1, 2", "1, 3"))
        if has_antlr:
            self.assertTrue(parse_and_verify_answer(r"\{1, 2\}", "1, 2"))
        else:
            self.assertFalse(parse_and_verify_answer(r"\{1, 2\}", "1, 2"))
        self.assertFalse(parse_and_verify_answer(r"\{1, 2\}", "1, 3"))

    def test_none_and_empty_inputs(self):
        from evaluation.verifiers.parser import (
            normalize_text_hybrid,
            parse_and_verify_answer,
        )
        self.assertFalse(parse_and_verify_answer(None, "4"))
        self.assertFalse(parse_and_verify_answer("4", None))
        self.assertFalse(parse_and_verify_answer("", ""))
        self.assertEqual(normalize_text_hybrid(""), "")

    def test_hybrid_parser_types(self):
        """sympy_parser_hybrid 返回类型契约：Rational/float/str/sympify 表达式。"""
        import sympy
        from evaluation.verifiers.parser import (
            sympy_parser_hybrid,
        )
        self.assertEqual(sympy_parser_hybrid("2/3"), sympy.Rational(2, 3))
        self.assertIsInstance(sympy_parser_hybrid(r"\$1,234.5"), float)
        self.assertEqual(sympy_parser_hybrid("2**3"), sympy.Integer(8))
        self.assertIsNone(sympy_parser_hybrid("not parseable ??"))


class TestTeacherForcing(unittest.TestCase):
    """Method 3（teacher-forcing，书 chF/02 的 3_teacher_forcing.py）。"""

    PROMPT = "Q\nA. 1\nB. 2\nC. 3\nD. 4\nAnswer: "

    def _rigged(self):
        return _RiggedLM(vocab_size=32, favored_token_id=2)   # chr('B')=66 -> 66%32=2

    def test_predicts_favored_letter_and_scores_shape(self):
        from inference.harness.mmlu import (
            predict_choice_teacher_forcing,
        )
        model = self._rigged()
        tokenizer = _MockCharTokenizer(vocab_size=32)
        prompt_fmt = torch.tensor(tokenizer.encode(self.PROMPT)).unsqueeze(0)

        pred, scores = predict_choice_teacher_forcing(model, tokenizer, prompt_fmt)
        self.assertEqual(pred, "B")
        self.assertEqual(set(scores), set("ABCD"))
        for value in scores.values():
            self.assertTrue(-1e6 < value < 0.0)     # logprob 必为负且有限

    def test_choices_path_scores_full_continuation(self):
        """choices 路径：continuation "X. <choice>" 逐 token teacher forcing。"""
        from inference.harness.mmlu import (
            avg_logprob_teacher_forced,
            predict_choice_teacher_forcing,
        )
        model = self._rigged()
        tokenizer = _MockCharTokenizer(vocab_size=32)
        prompt_fmt = torch.tensor(tokenizer.encode(self.PROMPT)).unsqueeze(0)
        choices = ["1", "2", "3", "4"]

        pred, scores = predict_choice_teacher_forcing(
            model, tokenizer, prompt_fmt, choices=choices,
        )
        self.assertEqual(pred, "B")
        self.assertEqual(set(scores), set("ABCD"))
        # rigged 模型分布与位置无关：多 token continuation 的平均 logprob
        # 应等于 favored token 的 logprob 与其他 token logprob 的混合，仍为负
        prompt_ids = tokenizer.encode(self.PROMPT)
        single = avg_logprob_teacher_forced(
            model, tokenizer, prompt_fmt, self.PROMPT, prompt_ids, "B", "2",
        )
        self.assertLess(single, 0.0)
        self.assertGreater(single, float("-inf"))

    def test_consistent_with_letter_logprob_method(self):
        """choices=None（单字母 continuation）时打分面与 Method 2 一致。"""
        from inference.harness.mmlu import (
            predict_choice_logprob,
            predict_choice_teacher_forcing,
        )
        torch.manual_seed(7)
        cfg = SimpleNamespace(
            vocab_size=32, context_length=64, embed_dim=32, d_ff=64,
            n_heads=4, n_layers=2, dropout=0.0, qkv_bias=True,
            dtype=torch.float32, is_train=False,
        )
        model = Model(cfg).eval()
        tokenizer = _MockCharTokenizer(vocab_size=32)
        prompt_fmt = torch.tensor(tokenizer.encode(self.PROMPT)).unsqueeze(0)

        _, tf_scores = predict_choice_teacher_forcing(model, tokenizer, prompt_fmt)
        _, lp_scores = predict_choice_logprob(model, tokenizer, prompt_fmt)
        for letter in "ABCD":
            self.assertAlmostEqual(tf_scores[letter], lp_scores[letter], places=5)
        # 同分则同预测
        self.assertEqual(
            max(tf_scores, key=tf_scores.get),
            max(lp_scores, key=lp_scores.get),
        )

    def test_evaluate_mmlu_with_teacher_forcing_predict_fn(self):
        """兼容 evaluate_mmlu 的 predict_fn(model, tokenizer, prompt_fmt) 约定。"""
        from inference.harness.mmlu import (
            evaluate_mmlu,
            predict_choice_teacher_forcing,
        )
        model = self._rigged()
        tokenizer = _MockCharTokenizer(vocab_size=32)
        data = [
            {"question": "1 + 1 = ?", "choices": ["1", "2", "3", "4"], "answer": 1},
            {"question": "2 * 3 = ?", "choices": ["5", "6", "7", "8"], "answer": "B"},
        ]
        results = evaluate_mmlu(
            model, tokenizer, data,
            predict_fn=lambda m, t, p: predict_choice_teacher_forcing(m, t, p)[0],
            verbose_every=0,
        )
        self.assertEqual(results["num_examples"], 2)
        self.assertEqual(results["num_correct"], 2)
        self.assertEqual(results["accuracy"], 1.0)

    def test_empty_continuation_raises(self):
        """continuation 编码被 prompt 公共前缀完全吞掉时应显式报错（书原版契约）。"""
        from inference.harness.mmlu import (
            avg_logprob_teacher_forced,
        )

        class _AbsorbingTokenizer:
            """encode 恒返回同一 id 序列（公共前缀 == 全长）的退化分词器。"""

            def encode(self, text):
                return [1, 2, 3]

            def decode(self, ids):
                return ""

        model = self._rigged()
        tokenizer = _AbsorbingTokenizer()
        prompt_fmt = torch.tensor([[1, 2, 3]])
        with self.assertRaises(ValueError):
            avg_logprob_teacher_forced(
                model, tokenizer, prompt_fmt, "Answer: ", [1, 2, 3], "A", "2",
            )


if __name__ == "__main__":
    unittest.main()
