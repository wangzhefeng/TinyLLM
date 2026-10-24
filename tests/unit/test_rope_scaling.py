"""RoPE NTK-aware / YaRN 外推测试（Phase 2 预训练×推断交叉）。"""
import math
import unittest

import torch

from layers.position_encoding.RoPE import (
    compute_rope,
    precompute_rope_params,
    yarn_attention_mscale,
)


class TestNTKScaling(unittest.TestCase):
    def test_ntk_lowers_effective_frequencies(self):
        """NTK-aware：theta 放大 → 最低频维的 inv_freq 更低（同位置旋转更慢）。"""
        head_dim, L = 32, 128
        cos_base, _ = precompute_rope_params(head_dim, context_length=L)
        cos_ntk, _ = precompute_rope_params(head_dim, context_length=L, ntk_factor=8.0)
        # 最高位置、最低频维（最后一维对应最低频）：NTK 下角度更小
        # cos(θ)：θ 越小 cos 越大
        self.assertGreater(cos_ntk[L - 1, -1].item(), cos_base[L - 1, -1].item())
        # 形状不变
        self.assertEqual(cos_ntk.shape, cos_base.shape)

    def test_ntk_usable_in_compute_rope(self):
        cos, sin = precompute_rope_params(32, context_length=128, ntk_factor=8.0)
        x = torch.randn(1, 4, 16, 32)
        out = compute_rope(x, cos, sin)
        self.assertEqual(out.shape, x.shape)
        self.assertTrue(torch.isfinite(out).all())


class TestYaRN(unittest.TestCase):
    def test_band_structure(self):
        """短波长高维不变，长波长低维除以 factor。"""
        head_dim = 16
        base_cos, _ = precompute_rope_params(head_dim, context_length=64)
        # 高频带边界需 > 最短波长（≈2π）：orig_ctx/beta_fast > 2π → beta_fast=4 得 8
        yarn_cos, _ = precompute_rope_params(
            head_dim, context_length=64,
            yarn_config={"original_context_length": 32, "factor": 4.0,
                         "beta_fast": 4, "beta_slow": 1},
        )
        # 最高频维（第 0 维，波长最短）应保持不变
        torch.testing.assert_close(yarn_cos[:, 0], base_cos[:, 0])
        # 整体有限且形状正确
        self.assertEqual(yarn_cos.shape, base_cos.shape)
        self.assertTrue(torch.isfinite(yarn_cos).all())
        # 低频维应有缩放（与基线不同）
        self.assertFalse(torch.allclose(yarn_cos[:, -1], base_cos[:, -1]))

    def test_mscale_formula(self):
        self.assertAlmostEqual(yarn_attention_mscale(4.0), 0.1 * math.log(4.0) + 1.0, places=6)
        self.assertAlmostEqual(yarn_attention_mscale(1.0), 1.0, places=6)


class TestRoPEModule(unittest.TestCase):
    def test_module_matches_functional(self):
        """模块化 RoPE 与函数式 precompute + compute 输出一致。"""
        from layers.position_encoding.RoPE import (
            RoPE,
            compute_rope_with_pos_ids,
        )

        torch.manual_seed(0)
        head_dim, L = 32, 128
        module = RoPE(head_dim=head_dim, theta_base=10_000.0, context_length=L)
        cos, sin = precompute_rope_params(head_dim, context_length=L)
        x = torch.randn(1, 4, 16, head_dim)
        torch.testing.assert_close(module(x), compute_rope(x, cos, sin))
        # 偏移应用（增量解码场景）
        torch.testing.assert_close(module(x, offset=3), compute_rope(x, cos, sin, offset=3))
        # pos_ids 版本（批量解码场景）
        pos_ids = torch.arange(16).unsqueeze(0).expand(1, 16)
        torch.testing.assert_close(
            module.forward_with_pos_ids(x, pos_ids),
            compute_rope_with_pos_ids(x, cos, sin, pos_ids),
        )

    def test_buffers_registered(self):
        """cos/sin 注册为 buffer：随模块移动、不进入 state_dict 持久化。"""
        from layers.position_encoding.RoPE import RoPE

        module = RoPE(head_dim=16, context_length=32)
        named = dict(module.named_buffers())
        self.assertIn("cos", named)
        self.assertIn("sin", named)
        self.assertEqual(named["cos"].shape, (32, 16))
        # persistent=False：不出现在 state_dict
        self.assertNotIn("cos", module.state_dict())


if __name__ == "__main__":
    unittest.main()
