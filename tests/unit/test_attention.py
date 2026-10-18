import unittest

import torch

from layers.attention import GroupedQueryAttention, MultiHeadAttention


class TestAttention(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)

    def test_multi_head_attention_shape(self):
        attn = MultiHeadAttention(
            d_model=32,
            d_out=32,
            n_heads=4,
            context_length=16,
            dropout=0.0,
        )
        x = torch.randn(2, 6, 32)
        y = attn(x)
        self.assertEqual(y.shape, (2, 6, 32))

    def test_multi_head_attention_is_causal(self):
        attn = MultiHeadAttention(
            d_model=32,
            d_out=32,
            n_heads=4,
            context_length=16,
            dropout=0.0,
        )
        attn.eval()
        x = torch.randn(1, 6, 32)
        y1 = attn(x)

        # Changing tokens at positions >= 3 must not change outputs at positions < 3.
        x2 = x.clone()
        x2[:, 3:, :] = torch.randn_like(x2[:, 3:, :])
        y2 = attn(x2)

        torch.testing.assert_close(y1[:, :3, :], y2[:, :3, :])

    def test_grouped_query_attention_shape(self):
        attn = GroupedQueryAttention(
            d_model=32,
            d_out=32,
            n_heads=4,
            context_length=16,
            num_kv_groups=2,
        )
        x = torch.randn(2, 6, 32)
        y = attn(x)
        self.assertEqual(y.shape, (2, 6, 32))


if __name__ == "__main__":
    unittest.main()
