"""Tiny Aya（Cohere，260M）tiny 单元测试（离线 CPU）。

覆盖：
- CAT 位置编码性质：零初始化仿射下平移保范数、位置 0 精确恒等、
  内积只依赖相对位置（纯旋转性质）、仿射参数可学习生效
- uvqk 注意力偏置：零初始化时为零贡献、扰动后确实改变输出、偏置形状正确
- tiny forward / backward（梯度可达 CAT 与 uvqk 参数）、weight tying
"""
import unittest
from types import SimpleNamespace

import torch

from models.aya.aya import (
    ComplexAffinePositionEncoding,
    GroupedQueryAttentionAya,
    Model,
)
from config.aya.model_cfgs import AYA_TINY_CONFIG


def _build_model():
    torch.manual_seed(0)
    return Model(SimpleNamespace(**dict(AYA_TINY_CONFIG)))


class TestComplexAffinePositionEncoding(unittest.TestCase):
    HEAD_DIM = 16

    def _cat(self):
        torch.manual_seed(0)
        return ComplexAffinePositionEncoding(
            head_dim=self.HEAD_DIM, theta_base=10_000.0, max_seq_len=64
        )

    def test_shapes(self):
        cat = self._cat()
        x = torch.randn(2, 3, 8, self.HEAD_DIM)
        self.assertEqual(cat(x).shape, x.shape)
        self.assertEqual(cat.cos.shape, (64, self.HEAD_DIM // 2))
        self.assertEqual(cat.scale.shape, (self.HEAD_DIM,))
        self.assertEqual(cat.shift.shape, (self.HEAD_DIM,))

    def test_norm_invariance_under_shift(self):
        """零初始化仿射下：同一 token 平移位置前后范数不变（纯旋转保范数）。"""
        cat = self._cat()
        x = torch.randn(2, 3, 8, self.HEAD_DIM)
        y0 = cat(x)              # 位置 0..7
        y1 = cat(x, offset=9)    # 同一批 token 平移到位置 9..16
        torch.testing.assert_close(y0.norm(dim=-1), y1.norm(dim=-1))
        # 且范数与原始输入一致
        torch.testing.assert_close(y0.norm(dim=-1), x.norm(dim=-1))
        torch.testing.assert_close(y1.norm(dim=-1), x.norm(dim=-1))

    def test_identity_at_zero_init_and_zero_position(self):
        """scale/shift 零初始化、位置 0（θ=0）时为精确恒等。"""
        cat = self._cat()
        x = torch.randn(2, 3, 1, self.HEAD_DIM)
        torch.testing.assert_close(cat(x), x)

    def test_relative_inner_product_invariance(self):
        """纯旋转性质：⟨q(t), k(t+s)⟩ 只依赖相对位置 s，与整体平移无关。"""
        cat = self._cat()
        q = torch.randn(1, 2, 6, self.HEAD_DIM)
        k = torch.randn(1, 2, 6, self.HEAD_DIM)
        for rel in (1, 3):
            base = (cat(q) * cat(k, offset=rel)).sum(dim=-1)
            for shift in (2, 5):
                shifted = (cat(q, offset=shift) * cat(k, offset=rel + shift)).sum(dim=-1)
                torch.testing.assert_close(base, shifted)

    def test_affine_parameters_take_effect(self):
        """仿射参数偏离零初始化后，输出不再等于纯旋转（可学习生效）。"""
        cat = self._cat()
        x = torch.randn(1, 2, 4, self.HEAD_DIM)
        rotation_only = cat(x)
        with torch.no_grad():
            cat.scale.fill_(0.5)
            cat.shift.fill_(0.25)
        affine = cat(x)
        self.assertFalse(torch.allclose(affine, rotation_only))
        expected = rotation_only * 1.5 + 0.25
        torch.testing.assert_close(affine, expected)

    def test_gradients_flow_to_affine(self):
        cat = self._cat()
        x = torch.randn(1, 2, 4, self.HEAD_DIM)
        cat(x).sum().backward()
        self.assertIsNotNone(cat.scale.grad)
        self.assertIsNotNone(cat.shift.grad)


class TestUvqkBias(unittest.TestCase):
    def _attn(self):
        torch.manual_seed(0)
        return GroupedQueryAttentionAya(
            d_model=32, n_heads=4, num_kv_groups=2, head_dim=8,
            theta_base=10_000.0, context_length=16,
        )

    def _mask(self, seq_len):
        return torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)

    def test_zero_init_equals_standard_gqa(self):
        """零初始化时，uvqk 偏置全部无效 → 精确等价于标准缩放点积 GQA。"""
        from layers.attention import _expand_grouped_kv

        attn = self._attn().eval()
        x = torch.randn(1, 6, 32)
        mask = self._mask(6)
        with torch.no_grad():
            out = attn(x, mask)
            # 手工复算：投影 → CAT → 点积 → causal softmax → out_proj
            q = attn.W_query(x).view(1, 6, 4, 8).transpose(1, 2)
            k = attn.W_key(x).view(1, 6, 2, 8).transpose(1, 2)
            v = attn.W_value(x).view(1, 6, 2, 8).transpose(1, 2)
            q, k = attn.cat(q), attn.cat(k)
            k = _expand_grouped_kv(k, attn.group_size, attn.n_heads)
            v = _expand_grouped_kv(v, attn.group_size, attn.n_heads)
            scores = q @ k.transpose(2, 3) * attn.scaling
            scores = scores.masked_fill(mask, -torch.inf)
            weights = torch.softmax(scores, dim=-1)
            ctx = (weights @ v).transpose(1, 2).reshape(1, 6, 32)
            reference = attn.out_proj(ctx)
        torch.testing.assert_close(out, reference)

    def test_bias_shapes(self):
        attn = self._attn()
        for name in ("q_bias", "k_bias", "u_bias", "v_bias"):
            self.assertEqual(getattr(attn, name).shape, (4, 8))
            self.assertTrue(torch.all(getattr(attn, name) == 0))

    def test_bias_changes_attention_output(self):
        """uvqk 偏置确实接入注意力：任一组扰动后输出改变。"""
        attn = self._attn().eval()
        x = torch.randn(1, 6, 32)
        mask = self._mask(6)
        with torch.no_grad():
            out_zero = attn(x, mask)
            for name in ("q_bias", "k_bias", "u_bias", "v_bias"):
                getattr(attn, name).fill_(0.5)
                out_perturbed = attn(x, mask)
                self.assertFalse(
                    torch.allclose(out_zero, out_perturbed),
                    f"perturbing {name} did not change the output",
                )
                getattr(attn, name).zero_()

    def test_gradients_flow_to_bias(self):
        attn = self._attn()
        x = torch.randn(1, 6, 32)
        attn(x, self._mask(6)).sum().backward()
        for name in ("q_bias", "k_bias", "u_bias", "v_bias"):
            # 注意：零初始化 + sum() 损失下偏置梯度可能整体为 0，
            # 但必须已建图（grad 张量存在）
            self.assertIsNotNone(getattr(attn, name).grad)


class TestAyaModel(unittest.TestCase):
    def test_forward_shape_and_finiteness(self):
        model = _build_model().eval()
        x = torch.randint(0, AYA_TINY_CONFIG["vocab_size"], (2, 10))
        with torch.no_grad():
            logits = model(x)
        self.assertEqual(logits.shape, (2, 10, AYA_TINY_CONFIG["vocab_size"]))
        self.assertTrue(torch.isfinite(logits).all())

    def test_backward(self):
        model = _build_model()
        x = torch.randint(0, AYA_TINY_CONFIG["vocab_size"], (2, 8))
        logits = model(x)
        logits.float().pow(2).mean().backward()
        for name, param in [
            ("tok_embed", model.tok_embed.weight),
            ("out_proj", model.blocks[0].attn.out_proj.weight),
            ("cat_scale", model.blocks[0].attn.cat.scale),
            ("cat_shift", model.blocks[0].attn.cat.shift),
            ("q_bias", model.blocks[0].attn.q_bias),
        ]:
            self.assertIsNotNone(param.grad, f"{name} has no grad")
            self.assertTrue(torch.isfinite(param.grad).all(), f"{name} grad not finite")

    def test_weight_tying(self):
        """weight tying：out_head.weight 就是 tok_embed.weight。"""
        model = _build_model()
        self.assertIs(model.out_head.weight, model.tok_embed.weight)

    def test_gqa_projection_shapes(self):
        model = _build_model()
        cfg = AYA_TINY_CONFIG
        attn = model.blocks[0].attn
        self.assertEqual(attn.W_query.out_features, cfg["n_heads"] * cfg["head_dim"])
        self.assertEqual(attn.W_key.out_features, cfg["n_kv_groups"] * cfg["head_dim"])
        self.assertEqual(attn.W_value.out_features, cfg["n_kv_groups"] * cfg["head_dim"])
        self.assertEqual(attn.group_size, cfg["n_heads"] // cfg["n_kv_groups"])

    def test_causal_mask_effect(self):
        """因果性：未来 token 改变不影响更早位置的 logits。"""
        model = _build_model().eval()
        x = torch.randint(0, AYA_TINY_CONFIG["vocab_size"], (1, 6))
        x2 = x.clone()
        x2[0, 4] = (x2[0, 4] + 1) % AYA_TINY_CONFIG["vocab_size"]
        with torch.no_grad():
            logits1 = model(x)
            logits2 = model(x2)
        torch.testing.assert_close(logits1[0, :4], logits2[0, :4])
        self.assertFalse(torch.allclose(logits1[0, 4:], logits2[0, 4:]))


if __name__ == "__main__":
    unittest.main()
