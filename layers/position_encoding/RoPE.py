# -*- coding: utf-8 -*-

# ***************************************************
# * File        : embedding.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-01-24
# * Version     : 1.0.012400
# * Description : Rotary Position Embedding
# * Link        : https://github.com/rasbt/LLMs-from-scratch/blob/2dc46bedc6e86b79a16c4099e557564cd23e03ef/ch02/03_bonus_embedding-vs-matmul/embeddings-and-linear-layers.ipynb
# *               https://www.k-a.in/pyt-rope.html
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

__all__ = []

# python libraries
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)

import torch
import torch.nn as nn

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]

# ==============================================================================
# RoPE implementation summary
#
#
# There are two common styles to implement RoPE, which are
# mathematically equivalent;
# they mainly differ in how the rotation matrix pairs dimensions.
#
# 1) Split-halves style (this repo, Hugging Face Transformers):
#
#   For hidden dim d = 8 (example):
#
#       [ x0   x1   x2   x3   x4   x5   x6   x7 ]
#         │    │    │    │    │    │    │    │
#         ▼    ▼    ▼    ▼    ▼    ▼    ▼    ▼
#        cos  cos  cos  cos  sin  sin  sin  sin
#
#   Rotation matrix:
#
#       [ cosθ   -sinθ    0      0   ... ]
#       [ sinθ    cosθ    0      0   ... ]
#       [  0       0    cosθ   -sinθ ... ]
#       [  0       0    sinθ    cosθ ... ]
#        ...
#
#   Here, the embedding dims are split into two halves and then
#   each one is rotated in blocks.
#
#
# 2) Interleaved (even/odd) style (original paper, Llama repo):
#
#   For hidden dim d = 8 (example):
#
#       [ x0   x1   x2   x3   x4   x5   x6   x7 ]
#         │    │    │    │    │    │    │    │
#         ▼    ▼    ▼    ▼    ▼    ▼    ▼    ▼
#        cos  sin  cos  sin  cos  sin  cos  sin
#
#   Rotation matrix:
#       [ cosθ  -sinθ    0      0   ... ]
#       [ sinθ   cosθ    0      0   ... ]
#       [  0      0    cosθ   -sinθ ... ]
#       [  0      0    sinθ    cosθ ... ]
#        ...
#
#   Here, embedding dims are interleaved as even/odd cosine/sine pairs.
#
# Both layouts encode the same relative positions; the only difference is how
# dimensions are paired.
# ==============================================================================


def precompute_rope_params(head_dim, theta_base=10_000, context_length=4096, freq_config=None, dtype=torch.float32):
    assert head_dim % 2 == 0, "Embedding dimension must be even"

    # Compute the inverse frequencies
    inv_freq = 1.0 / (theta_base ** (torch.arange(0, head_dim, 2, dtype=dtype)[: (head_dim // 2)].float() / head_dim))
    # Frequency adjustments
    if freq_config is not None:
        low_freq_wavelen = freq_config["original_context_length"] / freq_config["low_freq_factor"]
        high_freq_wavelen = freq_config["original_context_length"] / freq_config["high_freq_factor"]
        wavelen = 2 * torch.pi / inv_freq
        inv_freq_llama = torch.where(wavelen > low_freq_wavelen, inv_freq / freq_config["factor"], inv_freq)
        smooth_factor = (
            freq_config["original_context_length"] / wavelen - freq_config["low_freq_factor"]
        ) / (
            freq_config["high_freq_factor"] - freq_config["low_freq_factor"]
        )
        smoothed_inv_freq = (1 - smooth_factor) * (inv_freq / freq_config["factor"]) + smooth_factor * inv_freq
        is_medium_freq = (wavelen <= low_freq_wavelen) & (wavelen >= high_freq_wavelen)
        inv_freq_llama = torch.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)
        inv_freq = inv_freq_llama
    # Generate position indices
    positions = torch.arange(context_length, dtype=dtype)
    # Compute the angles, shape: (context_length, head_dim//2)
    angles = positions[:, None] * inv_freq[None, :]
    # Expand angles to match the head_dim, shape: (context_length, head_dim)
    angles = torch.cat([angles, angles], dim = 1)
    # Precompute sine and cosine
    cos = torch.cos(angles)
    sin = torch.sin(angles)

    return cos, sin


def compute_rope(x, cos, sin, offset=0):
    """
    RoPE: RoFormer: Enhanced Transformer with Rotary Position Embedding (2021)
    https://arxiv.org/abs/2104.09864
    """
    # x: (batch_size, n_heads, seq_len, head_dim)
    batch_size, n_heads, seq_len, head_dim = x.shape
    assert head_dim % 2 == 0, "Head dimension must be even"

    # Split x into first half and second half
    x1 = x[..., : head_dim // 2]  # First half
    x2 = x[..., head_dim // 2 :]  # Second half

    # Adjust sin and cos shapes, Shape: (1, 1, seq_len, head_dim)
    cos = cos[offset:(offset+seq_len), :].unsqueeze(0).unsqueeze(0)
    sin = sin[offset:(offset+seq_len), :].unsqueeze(0).unsqueeze(0)

    # Apply the rotary transformation
    rotated = torch.cat((-x2, x1), dim=-1)
    x_rotated = (x * cos) + (rotated * sin)
    # It's ok to use lower-precision after applying cos and sin rotation
    return x_rotated.to(dtype=x.dtype)


def compute_rope_with_pos_ids(x, cos, sin, pos_ids):
    B, H, L, D = x.shape
    cos_sel = cos[pos_ids]  # (B, L, D)
    sin_sel = sin[pos_ids]  # (B, L, D)
    cos_sel = cos_sel.unsqueeze(1)  # (B, 1, L, D)
    sin_sel = sin_sel.unsqueeze(1)  # (B, 1, L, D)
    x1 = x[..., : D // 2]
    x2 = x[..., D // 2:]
    rotated = torch.cat((-x2, x1), dim=-1)
    x_rotated = (x * cos_sel) + (rotated * sin_sel)
    return x_rotated.to(dtype=x.dtype)


# TODO
class RoPE(nn.Module):
    
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        pass




# 测试代码 main 函数
def main():
    # Settings
    batch_size = 2
    n_heads = 4
    head_dim = 16

    llama2_context_len = 4096
    llama2_theta_base = 10_000

    llama3_context_len = 8192
    llama3_theta_base = 500_000

    # ------------------------------
    # llama 2
    # ------------------------------
    # Instantiate RoPE parameters
    cos, sin = precompute_rope_params(
        head_dim = head_dim, 
        theta_base = llama2_theta_base,
        context_length = llama2_context_len,
    )
    # Dummy query and key tensors
    torch.manual_seed(123)
    queries = torch.randn(batch_size, n_heads, llama2_context_len, head_dim)
    keys = torch.randn(batch_size, n_heads, llama2_context_len, head_dim)
    # Apply rotary position embeddings
    queries_rot = compute_rope(queries, cos, sin)
    keys_rot = compute_rope(keys, cos, sin)
    # ------------------------------
    # llama 3
    # ------------------------------
    # Instantiate RoPE parameters
    cos, sin = precompute_rope_params(
        head_dim = head_dim, 
        theta_base = llama3_theta_base,
        context_length = llama3_context_len,
    )
    # Dummy query and key tensors
    torch.manual_seed(123)
    queries = torch.randn(batch_size, n_heads, llama3_context_len, head_dim)
    keys = torch.randn(batch_size, n_heads, llama3_context_len, head_dim)
    # Apply rotary position embeddings
    queries_rot = compute_rope(queries, cos, sin)
    keys_rot = compute_rope(keys, cos, sin)

if __name__ == "__main__":
    main()
