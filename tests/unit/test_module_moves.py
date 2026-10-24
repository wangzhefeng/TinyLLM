import importlib
import unittest


class TestCompatibilityShims(unittest.TestCase):
    def test_old_and_new_paths_share_implementation(self):
        pairs = [
            ("inference.loaders.qwen3.load_weights", "models.io.qwen3.load_weights"),
            ("post_training.reasoning.inference.voting_and_verification", "inference.strategies.voting_and_verification"),
            ("post_training.reasoning.evaluation.benchmark_based_eval.math_verifiers", "inference.evaluation.benchmarks.math_verifiers"),
            ("post_training.reasoning.evaluation.judgement_based_eval.llm_judges", "inference.evaluation.judges.llm_judges"),
            ("post_training.reasoning.training.reinforcement_learning", "post_training.rl.core", "generate_group_samples"),
            ("post_training.common.data_provider.instruction_format", "layers.tokenizers.chat", "format_input_alpaca"),
        ]
        for old, new, *attribute in pairs:
            with self.subTest(old=old):
                old_module = importlib.import_module(old)
                new_module = importlib.import_module(new)
                if attribute:
                    self.assertIs(getattr(old_module, attribute[0]), getattr(new_module, attribute[0]))
                else:
                    self.assertIs(old_module, new_module)


if __name__ == "__main__":
    unittest.main()
