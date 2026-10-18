import unittest
from types import SimpleNamespace

import torch

from layers.feed_forward import (
    FeedForwardGELU,
    FeedForwardReLU,
    FeedForwardSiLU,
)


class TestFeedForward(unittest.TestCase):
    def setUp(self):
        self.cfg = SimpleNamespace(
            embed_dim=32,
            d_ff=64,
            dtype=torch.float32,
        )
        self.x = torch.randn(2, 5, 32)

    def test_relu_shape(self):
        layer = FeedForwardReLU(self.cfg)
        self.assertEqual(layer(self.x).shape, self.x.shape)

    def test_gelu_shape(self):
        layer = FeedForwardGELU(self.cfg)
        self.assertEqual(layer(self.x).shape, self.x.shape)

    def test_silu_swiglu_shape(self):
        # FeedForwardSiLU is the SwiGLU variant used by modern LLMs.
        layer = FeedForwardSiLU(self.cfg)
        self.assertEqual(layer(self.x).shape, self.x.shape)


if __name__ == "__main__":
    unittest.main()
