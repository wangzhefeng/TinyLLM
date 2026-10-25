"""NF4 量化原语测试（QLoRA，https://arxiv.org/abs/2305.14314）。"""
import unittest

import torch

from layers.quantization.nf4 import (
    NF4_CODEBOOK,
    nf4_quantize,
    nf4_dequantize,
    quantize_scales,
    dequantize_scales,
    nf4_double_quantize,
    nf4_double_dequantize,
)


class TestCodebook(unittest.TestCase):
    def test_codebook_shape_and_range(self):
        """16 个码本点，单调递增，端点为 ±1，含 0。"""
        self.assertEqual(NF4_CODEBOOK.shape, (16,))
        self.assertAlmostEqual(float(NF4_CODEBOOK[0]), -1.0)
        self.assertAlmostEqual(float(NF4_CODEBOOK[-1]), 1.0)
        self.assertEqual(float(NF4_CODEBOOK[7]), 0.0)
        self.assertTrue((NF4_CODEBOOK[1:] > NF4_CODEBOOK[:-1]).all())


class TestQuantizeRoundtrip(unittest.TestCase):
    def test_roundtrip_error_bounded(self):
        """对 N(0,1) 张量，量化往返的相对误差有界（NF4 对正态数据最优）。"""
        torch.manual_seed(0)
        x = torch.randn(64, 128)
        idx, scales, shape, bs = nf4_quantize(x)
        x_hat = nf4_dequantize(idx, scales, shape, bs)
        rel_err = (x - x_hat).norm() / x.norm()
        # NF4 对正态数据的典型相对误差约 5-8%，这里放宽松阈值保证稳定
        self.assertLess(float(rel_err), 0.12)
        self.assertTrue(torch.isfinite(x_hat).all())

    def test_shapes_and_dtypes(self):
        """indices 为 uint8 且与输入同形状；scales 为每块一个 fp32。"""
        x = torch.randn(3, 100)          # 300 元素 → ceil(300/64)=5 块
        idx, scales, shape, bs = nf4_quantize(x, block_size=64)
        self.assertEqual(idx.dtype, torch.uint8)
        self.assertEqual(idx.shape, x.shape)
        self.assertEqual(shape, x.shape)
        self.assertEqual(scales.dtype, torch.float32)
        self.assertEqual(scales.numel(), 5)
        self.assertEqual(bs, 64)
        # 反量化还原形状
        x_hat = nf4_dequantize(idx, scales, shape, bs)
        self.assertEqual(x_hat.shape, x.shape)

    def test_indices_within_codebook(self):
        x = torch.randn(2, 64) * 10
        idx, _, _, _ = nf4_quantize(x)
        self.assertTrue((idx >= 0).all() and (idx < 16).all())

    def test_non_multiple_block_size(self):
        """元素数不是 block_size 的倍数时，padding 处理正确。"""
        x = torch.randn(37)              # 37 < 64
        idx, scales, shape, bs = nf4_quantize(x, block_size=64)
        self.assertEqual(scales.numel(), 1)
        x_hat = nf4_dequantize(idx, scales, shape, bs)
        self.assertEqual(x_hat.shape, x.shape)
        self.assertTrue(torch.isfinite(x_hat).all())

    def test_zero_and_constant_blocks_no_nan(self):
        """全零块与全同值块：量化不除零、反量化不产生 NaN。"""
        x = torch.zeros(1, 128)
        idx, scales, shape, bs = nf4_quantize(x)
        x_hat = nf4_dequantize(idx, scales, shape, bs)
        self.assertTrue(torch.isfinite(x_hat).all())
        self.assertTrue((x_hat == 0).all())

        x = torch.full((1, 64), 3.14)
        idx, scales, shape, bs = nf4_quantize(x)
        x_hat = nf4_dequantize(idx, scales, shape, bs)
        self.assertTrue(torch.isfinite(x_hat).all())
        # 全同值块：所有元素映射到最近码本点 3.14→scale·1.0，应接近原值
        self.assertTrue(((x_hat - 3.14).abs() < 0.5).all())

    def test_storage_reduction(self):
        """量化存储量小于 fp32 原始存储（教学验证：4bit + 每块 scale）。"""
        x = torch.randn(4, 256)          # 1024 元素
        idx, scales, _, _ = nf4_quantize(x)
        orig_bytes = x.numel() * 4
        quant_bytes = idx.numel() * 1 + scales.numel() * 4
        self.assertLess(quant_bytes, orig_bytes)


class TestDoubleQuantization(unittest.TestCase):
    def test_scale_roundtrip(self):
        """对 scales 的二次量化往返误差很小（uint8 + 组 absmax）。"""
        torch.manual_seed(0)
        scales = torch.rand(300) * 3.0
        idx, gmax, n = quantize_scales(scales, group_size=64)
        rec = dequantize_scales(idx, gmax, n, group_size=64)
        rel = (scales - rec).norm() / scales.norm()
        self.assertLess(float(rel), 0.01)

    def test_double_quant_roundtrip_error_bounded(self):
        """double quantization 后 NF4 往返误差仍然有界（比单次略差）。"""
        torch.manual_seed(0)
        x = torch.randn(64, 128)
        idx, sidx, gmax, shape, bs, gs = nf4_double_quantize(x)
        x_hat = nf4_double_dequantize(idx, sidx, gmax, shape, bs, gs)
        rel_err = (x - x_hat).norm() / x.norm()
        self.assertLess(float(rel_err), 0.13)
        self.assertTrue(torch.isfinite(x_hat).all())

    def test_storage_beats_single_quant(self):
        """double quantization 存储量小于单次量化（≈4.127 < 4.5 bit/参数）。"""
        x = torch.randn(4, 256)          # 1024 元素 -> 16 块
        idx, scales, _, _ = nf4_quantize(x)
        _, sidx, gmax, _, _, _ = nf4_double_quantize(x)
        single_bytes = idx.numel() * 1 + scales.numel() * 4
        double_bytes = idx.numel() * 1 + sidx.numel() * 1 + gmax.numel() * 4
        self.assertLess(double_bytes, single_bytes)


if __name__ == "__main__":
    unittest.main()
