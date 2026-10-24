"""Gemma 4 tiny 单元测试（离线 CPU）。

覆盖：
- layer_pattern 交错比参数化（与官方 E2B/E4B layer_types 逐位对照）
- proportional partial RoPE 的恒等尾部 / 位置 0 恒等
- tiny forward 形状与有限性、backward、weight tying、逐层 head_dim、logit softcap
"""
import unittest
from types import SimpleNamespace

import torch

from models.gemma4.gemma4 import (
    Model,
    layer_pattern,
    precompute_proportional_rope_params,
)
from config.gemma4.model_cfgs import GEMMA4_TINY_CONFIG


def _build_model():
    torch.manual_seed(0)
    return Model(SimpleNamespace(**dict(GEMMA4_TINY_CONFIG)))


class TestLayerPattern(unittest.TestCase):
    def test_e2b_official_pattern(self):
        """官方 E2B layer_types：35 层 4:1，full 落在 4,9,...,34。"""
        pattern = layer_pattern(35, local_per_global=4)
        full_idx = [i for i, t in enumerate(pattern) if t == "full_attention"]
        self.assertEqual(full_idx, [4, 9, 14, 19, 24, 29, 34])
        self.assertEqual(pattern[-1], "full_attention")

    def test_e4b_official_pattern(self):
        """官方 E4B layer_types：42 层 5:1，full 落在 5,11,...,41。"""
        pattern = layer_pattern(42, local_per_global=5)
        full_idx = [i for i, t in enumerate(pattern) if t == "full_attention"]
        self.assertEqual(full_idx, [5, 11, 17, 23, 29, 35, 41])
        self.assertEqual(pattern[-1], "full_attention")

    def test_ratio(self):
        pattern = layer_pattern(30, local_per_global=4)
        n_local = pattern.count("sliding_attention")
        n_global = pattern.count("full_attention")
        self.assertEqual(n_local / n_global, 4.0)
        self.assertEqual(len(pattern), 30)

    def test_default_is_five_to_one(self):
        """transformers 默认 5:1。"""
        pattern = layer_pattern(12)
        full_idx = [i for i, t in enumerate(pattern) if t == "full_attention"]
        self.assertEqual(full_idx, [5, 11])

    def test_last_layer_forced_full(self):
        pattern = layer_pattern(7, local_per_global=5)
        self.assertEqual(pattern[-1], "full_attention")


class TestProportionalRoPE(unittest.TestCase):
    HEAD_DIM = 512
    FACTOR = 0.25
    N_ROT = int(FACTOR * HEAD_DIM // 2)  # 旋转频率对数 = 64

    def _rope(self, context_length=16):
        return precompute_proportional_rope_params(
            head_dim=self.HEAD_DIM,
            theta_base=1_000_000.0,
            partial_rotary_factor=self.FACTOR,
            context_length=context_length,
        )

    def test_shapes(self):
        cos, sin = self._rope()
        self.assertEqual(cos.shape, (16, self.HEAD_DIM))
        self.assertEqual(sin.shape, (16, self.HEAD_DIM))

    def test_identity_tail(self):
        """未旋转的 75% 频率对：cos=1、sin=0（恒等旋转）。"""
        cos, sin = self._rope()
        half = self.HEAD_DIM // 2
        torch.testing.assert_close(
            cos[:, self.N_ROT:half], torch.ones(16, half - self.N_ROT)
        )
        torch.testing.assert_close(
            sin[:, self.N_ROT:half], torch.zeros(16, half - self.N_ROT)
        )
        # 旋转部分在位置 > 0 处确有非平凡角度
        self.assertGreater(sin[1:, : self.N_ROT].abs().max().item(), 0.0)

    def test_position_zero_is_identity(self):
        """位置 0：全部 cos=1、sin=0。"""
        cos, sin = self._rope()
        torch.testing.assert_close(cos[0], torch.ones(self.HEAD_DIM))
        torch.testing.assert_close(sin[0], torch.zeros(self.HEAD_DIM))

    def test_rotation_preserves_norm(self):
        """旋转（含恒等尾部）保范数：任意输入经 cos/sin 变换范数不变。"""
        cos, sin = self._rope()
        x = torch.randn(2, 3, 16, self.HEAD_DIM)
        x1, x2 = x[..., : self.HEAD_DIM // 2], x[..., self.HEAD_DIM // 2:]
        rotated = torch.cat((-x2, x1), dim=-1)
        y = x * cos + rotated * sin
        torch.testing.assert_close(y.norm(dim=-1), x.norm(dim=-1))


class TestGemma4Model(unittest.TestCase):
    def test_forward_shape_and_finiteness(self):
        model = _build_model().eval()
        x = torch.randint(0, GEMMA4_TINY_CONFIG["vocab_size"], (2, 10))
        with torch.no_grad():
            logits = model(x)
        self.assertEqual(logits.shape, (2, 10, GEMMA4_TINY_CONFIG["vocab_size"]))
        self.assertTrue(torch.isfinite(logits).all())

    def test_backward(self):
        model = _build_model()
        x = torch.randint(0, GEMMA4_TINY_CONFIG["vocab_size"], (2, 8))
        logits = model(x)
        logits.sum().backward()
        for name, param in [("tok_embed", model.tok_embed.weight),
                            ("out_proj", model.blocks[0].attn.out_proj.weight),
                            ("ff", model.blocks[0].ff.fc3.weight)]:
            self.assertIsNotNone(param.grad, f"{name} has no grad")
            self.assertTrue(torch.isfinite(param.grad).all(), f"{name} grad not finite")

    def test_weight_tying(self):
        model = _build_model()
        self.assertIs(model.out_head.weight, model.tok_embed.weight)

    def test_layer_types_generated_in_model(self):
        """cfg 未给 layer_types 时按 local_per_global 生成：6 层 2:1。"""
        model = _build_model()
        self.assertEqual(
            model.layer_types,
            ["sliding_attention", "sliding_attention", "full_attention",
             "sliding_attention", "sliding_attention", "full_attention"],
        )

    def test_per_layer_head_dim(self):
        """full 层用 global_head_dim=32，sliding 层用 head_dim=16。"""
        model = _build_model()
        cfg = GEMMA4_TINY_CONFIG
        for attn_type, block in zip(model.layer_types, model.blocks):
            expected = (cfg["n_heads"] * (cfg["global_head_dim"]
                         if attn_type == "full_attention" else cfg["head_dim"]))
            self.assertEqual(block.attn.W_query.out_features, expected)
            self.assertEqual(block.attn.head_dim,
                             cfg["global_head_dim"] if attn_type == "full_attention"
                             else cfg["head_dim"])

    def test_final_logit_softcap(self):
        """tanh 软限幅：|logits| < softcap。"""
        model = _build_model().eval()
        x = torch.randint(0, GEMMA4_TINY_CONFIG["vocab_size"], (1, 12))
        with torch.no_grad():
            logits = model(x)
        self.assertLess(logits.abs().max().item(), GEMMA4_TINY_CONFIG["final_logit_softcapping"])


if __name__ == "__main__":
    unittest.main()
