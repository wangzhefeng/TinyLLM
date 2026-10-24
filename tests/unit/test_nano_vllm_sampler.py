import unittest

import torch

from inference.serving.nano_vllm.layers.sampler import Sampler


class TestSamplerTopK(unittest.TestCase):
    def test_variable_top_k_support(self):
        torch.manual_seed(0)
        logits = torch.tensor([
            [0.1, 3.0, 2.0, 1.0],
            [0.1, 3.0, 2.0, 1.0],
            [0.1, 3.0, 2.0, 1.0],
        ])
        tokens = Sampler()(logits, torch.ones(3), torch.tensor([1, 2, -1]))
        self.assertEqual(tokens[0].item(), 1)
        self.assertIn(tokens[1].item(), {1, 2})
        self.assertGreaterEqual(tokens[2].item(), 0)

    def test_greedy_rows_are_unchanged(self):
        logits = torch.randn(3, 20)
        tokens = Sampler()(logits, torch.zeros(3), torch.tensor([1, 2, 3]))
        self.assertTrue(torch.equal(tokens, logits.argmax(dim=-1)))


if __name__ == "__main__":
    unittest.main()
