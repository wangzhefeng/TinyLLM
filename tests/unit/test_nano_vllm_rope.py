# -*- coding: utf-8 -*-

# ***************************************************
# * File        : test_nano_vllm_rope.py
# * Description : nano_vllm RoPE 半分布局(half-split)回归测试：
# *               与 layers/position_encoding/RoPE.py（HF Qwen3 风格）在
# *               随机张量上逐元素等价；并断言 cos/sin 缓存为 [f, f] 拼接布局。
# * Requirement : torch >= 2.0
# ***************************************************

import unittest

import torch

from layers.position_encoding.RoPE import (
    compute_rope_with_pos_ids,
    precompute_rope_params,
)
from inference.serving.nano_vllm.layers.rotary_embedding import (
    RotaryEmbedding,
    get_rope,
)


class TestNanoVllmRoPEHalfSplit(unittest.TestCase):

    def test_matches_shared_rope_on_random_tensors(self):
        """随机 q/k + 随机位置：nano_vllm RoPE 与共享 RoPE 实现输出一致"""
        torch.manual_seed(0)
        head_dim, max_pos, base = 32, 128, 10000.0
        num_tokens, n_heads, n_kv_heads = 17, 4, 2
        rope = RotaryEmbedding(head_dim, head_dim, max_pos, base)
        cos_ref, sin_ref = precompute_rope_params(
            head_dim, theta_base=base, context_length=max_pos,
        )
        # 随机乱序位置（覆盖非连续位置索引）
        positions = torch.randperm(max_pos)[:num_tokens]
        q = torch.randn(num_tokens, n_heads, head_dim)
        k = torch.randn(num_tokens, n_kv_heads, head_dim)

        q_out, k_out = rope(positions, q.clone(), k.clone())

        # 参考实现输入为 (B, H, L, D)：把 nano_vllm 的 token 维视作 L
        pos_ids = positions.unsqueeze(0)  # (1, num_tokens)
        q_ref = compute_rope_with_pos_ids(
            q.transpose(0, 1).unsqueeze(0), cos_ref, sin_ref, pos_ids,
        ).squeeze(0).transpose(0, 1)
        k_ref = compute_rope_with_pos_ids(
            k.transpose(0, 1).unsqueeze(0), cos_ref, sin_ref, pos_ids,
        ).squeeze(0).transpose(0, 1)

        torch.testing.assert_close(q_out, q_ref)
        torch.testing.assert_close(k_out, k_ref)

    def test_cos_sin_buffers_use_half_split_layout(self):
        """缓存布局必须是 [f, f] 拼接（维度 i 与 i + d/2 配对），
        而非交错 repeat_interleave（维度 2i 与 2i+1 配对）"""
        rope = RotaryEmbedding(16, 16, 64, 10000.0)
        half = 8
        # 前半与后半完全相同 -> half-split
        torch.testing.assert_close(rope.cos[:, :half], rope.cos[:, half:])
        torch.testing.assert_close(rope.sin[:, :half], rope.sin[:, half:])
        # 且相邻两维频率不同（排除 repeat_interleave 交错布局）
        self.assertFalse(torch.allclose(rope.cos[:, 0], rope.cos[:, 1]))

    def test_rotation_preserves_norm(self):
        """旋转是正交变换：每个 (x_i, x_{i+d/2}) 对的模长不变"""
        torch.manual_seed(1)
        head_dim = 16
        rope = RotaryEmbedding(head_dim, head_dim, 64, 10000.0)
        positions = torch.arange(5)
        q = torch.randn(5, 2, head_dim)
        k = torch.randn(5, 2, head_dim)
        q_out, _ = rope(positions, q.clone(), k.clone())
        half = head_dim // 2
        pair_norm = lambda x: (x[..., :half] ** 2 + x[..., half:] ** 2).sum(-1)
        torch.testing.assert_close(pair_norm(q_out), pair_norm(q))

    def test_get_rope_caches_instance(self):
        self.assertIs(
            get_rope(16, 16, 64, 10000.0),
            get_rope(16, 16, 64, 10000.0),
        )


if __name__ == "__main__":
    unittest.main()
