"""LLM-as-a-Judge 测试：rubric prompt / 抽分 / 汇总 / 后端注入（reasoning chF/04）。"""
import unittest


class TestJudgePrompt(unittest.TestCase):
    def test_rubric_and_sections_present(self):
        from inference.evaluation.judges.llm_judges import (
            build_judge_prompt,
        )
        prompt = build_judge_prompt(
            question="What is 2+2?",
            response="4",
            reference_answer="4",
        )
        # 书 chF/04 的 rubric 五档标准
        for level in ("1:", "2:", "3:", "4:", "5:"):
            self.assertIn(level, prompt)
        self.assertIn("Instruction:\nWhat is 2+2?", prompt)
        self.assertIn("Reference Answer:\n4", prompt)
        self.assertIn("Answer:\n4", prompt)
        self.assertTrue(prompt.endswith("Evaluation: "))

    def test_without_reference_answer(self):
        from inference.evaluation.judges.llm_judges import (
            build_judge_prompt,
        )
        prompt = build_judge_prompt(question="Q?", response="R")
        self.assertNotIn("Reference Answer:", prompt)
        self.assertIn("Instruction:\nQ?", prompt)
        self.assertIn("Answer:\nR", prompt)


class TestParseScore(unittest.TestCase):
    def test_common_formats(self):
        from inference.evaluation.judges.llm_judges import (
            parse_score,
        )
        self.assertEqual(parse_score("Rating: 4"), 4)
        self.assertEqual(parse_score("rating: 4"), 4)
        self.assertEqual(parse_score("4/5"), 4)
        self.assertEqual(parse_score("The score is 5."), 5)
        self.assertEqual(parse_score("Evaluation: 1 - bad"), 1)

    def test_default_on_no_match(self):
        from inference.evaluation.judges.llm_judges import (
            parse_score,
        )
        self.assertEqual(parse_score("no digit here"), 3)          # 书原版默认
        self.assertEqual(parse_score("no digit here", default=2), 2)
        self.assertEqual(parse_score("", default=1), 1)

    def test_score_range_restricted(self):
        """锁定书原版正则的边界语义（防止回归）：

        - "6/10"：'1' 后跟数字 '0' 不满足 (?:\\D|$)，无匹配 -> default 3
        - "10 out of 10"：同理无匹配 -> default 3
        """
        from inference.evaluation.judges.llm_judges import (
            parse_score,
        )
        self.assertEqual(parse_score("6/10"), 3)
        self.assertEqual(parse_score("10 out of 10"), 3)


class TestSummarizeScores(unittest.TestCase):
    def test_summary_stats(self):
        from inference.evaluation.judges.llm_judges import (
            summarize_scores,
        )
        s = summarize_scores([4, 5, 3, 4])
        self.assertEqual(s["total"], 4)
        self.assertEqual(s["mean"], 4.0)
        self.assertEqual(s["median"], 4.0)
        self.assertEqual(s["counts"][3], 1)
        self.assertEqual(s["counts"][4], 2)
        self.assertEqual(s["counts"][5], 1)
        self.assertEqual(sum(s["counts"].values()), 4)

    def test_empty_scores(self):
        from inference.evaluation.judges.llm_judges import (
            summarize_scores,
        )
        s = summarize_scores([])
        self.assertEqual(s["total"], 0)
        self.assertEqual(s["mean"], 0.0)


class TestEvaluateResponse(unittest.TestCase):
    def test_full_flow_with_mock_backend(self):
        from inference.evaluation.judges.llm_judges import (
            evaluate_response,
        )
        mock = lambda prompt: "Mostly correct.\nRating: 4"  # noqa: E731
        result = evaluate_response(
            question="Q?", response="R", query_fn=mock, reference_answer="REF",
        )
        self.assertEqual(result["score"], 4)
        self.assertIn("Rating: 4", result["judge_text"])
        self.assertIn("Q?", result["prompt"])

    def test_query_fn_required(self):
        from inference.evaluation.judges.llm_judges import (
            evaluate_response,
        )
        with self.assertRaises(ValueError):
            evaluate_response(question="Q?", response="R", query_fn=None)


class TestOllamaBackend(unittest.TestCase):
    def test_query_ollama_clear_error_when_unreachable(self):
        """本地无 Ollama 时应抛出带排查建议的 RuntimeError（不联网即可测）。"""
        from inference.evaluation.judges.llm_judges_Ollama_API import (
            query_ollama,
        )
        with self.assertRaises(RuntimeError) as ctx:
            query_ollama("ping", url="http://127.0.0.1:1/api/generate", timeout=2)
        self.assertIn("Ollama", str(ctx.exception))

    def test_module_import_has_no_side_effects(self):
        """import 不应触发任何网络调用或模型加载。"""
        import importlib

        import inference.evaluation.judges.llm_judges_Ollama_API as m
        importlib.reload(m)      # 若模块级有副作用，此处即会执行网络/加载


class TestQwen3Backend(unittest.TestCase):
    def test_module_import_has_no_side_effects(self):
        """import 不得拉起 transformers pipeline（延迟到函数内）。"""
        import inference.evaluation.judges.llm_judges_Qwen3 as m
        self.assertIsNone(m._QWEN3_PIPELINE)


if __name__ == "__main__":
    unittest.main()
