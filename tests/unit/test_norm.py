import unittest

import torch

from layers.normalization.layer_norm import LayerNorm, LayerNormPyTorch
from layers.normalization.rms_norm import RMSNorm


class TestNormalization(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)
        self.x = torch.randn(4, 8, 16)

    def test_layer_norm_shape_and_statistics(self):
        norm = LayerNorm(embed_dim=16)
        y = norm(self.x)
        self.assertEqual(y.shape, self.x.shape)
        mean = y.mean(dim=-1)
        var = y.var(dim=-1, unbiased=False)
        torch.testing.assert_close(mean, torch.zeros_like(mean), atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(var, torch.ones_like(var), atol=1e-4, rtol=1e-4)

    def test_layer_norm_matches_pytorch(self):
        ours = LayerNorm(embed_dim=16)
        ref = LayerNormPyTorch(embed_dim=16)
        # Copy affine parameters so the comparison is meaningful.
        ours.scale.data.copy_(ref.ln.weight.data)
        ours.shift.data.copy_(ref.ln.bias.data)
        torch.testing.assert_close(ours(self.x), ref(self.x), atol=1e-6, rtol=1e-6)

    def test_rms_norm_shape(self):
        norm = RMSNorm(embed_dim=16)
        y = norm(self.x)
        self.assertEqual(y.shape, self.x.shape)
        self.assertTrue(torch.isfinite(y).all())


if __name__ == "__main__":
    unittest.main()
