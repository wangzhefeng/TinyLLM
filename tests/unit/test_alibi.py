"""ALiBi 位置编码测试（Phase 2 预训练域）。"""
import math
import unittest

import torch

from layers.position_encoding.ALiBi import build_alibi_bias, get_alibi_slopes


class TestAlibiSlopes(unittest.TestCase):
    def test_power_of_two_formula(self):
        slopes = get_alibi_slopes(8)
        expected = [2 ** (-8 * i / 8) for i in range(1, 9)]  # 1/2 .. 1/256
        torch.testing.assert_close(slopes, torch.tensor(expected, dtype=torch.float32))

    def test_non_power_of_two(self):
        """非二次幂头数：前 closest 个是标准二次幂斜率，其余为几何插值（论文附录 B）。"""
        slopes = get_alibi_slopes(12)
        self.assertEqual(slopes.shape, (12,))
        self.assertTrue((slopes > 0).all())
        # 前 8 个 = power_of_two_slopes(8)
        torch.testing.assert_close(slopes[:8], get_alibi_slopes(8))
        # 额外 4 个 = slopes(16) 的偶数下标
        extra_expected = get_alibi_slopes(16)[0::2][:4]
        torch.testing.assert_close(slopes[8:], extra_expected)


class TestAlibiBias(unittest.TestCase):
    def test_values_and_causality(self):
        bias = build_alibi_bias(n_heads=2, seq_len=4)
        self.assertEqual(bias.shape, (1, 2, 4, 4))
        s0, s1 = get_alibi_slopes(2).tolist()
        # i=3, j=1：距离 2 → bias = -slope * 2
        self.assertAlmostEqual(bias[0, 0, 3, 1].item(), -s0 * 2, places=5)
        self.assertAlmostEqual(bias[0, 1, 3, 1].item(), -s1 * 2, places=5)
        # 自身距离 0 → 偏置 0
        self.assertAlmostEqual(bias[0, 0, 2, 2].item(), 0.0, places=5)
        # 未来位置 -inf（causal）
        self.assertTrue(torch.isinf(bias[0, 0, 0, 1]))

    def test_non_causal_has_no_inf(self):
        bias = build_alibi_bias(n_heads=2, seq_len=4, causal=False)
        self.assertTrue(torch.isfinite(bias).all())

    def test_usable_with_sdpa(self):
        """ALiBi 偏置可直接作为 SDPA 的 additive mask。"""
        q = torch.randn(1, 2, 4, 8)
        k = torch.randn(1, 2, 4, 8)
        v = torch.randn(1, 2, 4, 8)
        bias = build_alibi_bias(n_heads=2, seq_len=4)
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        self.assertEqual(out.shape, (1, 2, 4, 8))
        self.assertTrue(torch.isfinite(out).all())

    def test_extrapolates_beyond_training_len(self):
        """外推性质：更长序列的偏置就是同一线性函数的延伸（无参数、无预计算表）。"""
        bias4 = build_alibi_bias(2, 4)
        bias8 = build_alibi_bias(2, 8)
        torch.testing.assert_close(bias8[..., :4, :4], bias4)


if __name__ == "__main__":
    unittest.main()
