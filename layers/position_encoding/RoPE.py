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

__all__ = [
    "yarn_attention_mscale",
    "precompute_rope_params",
    "compute_rope",
    "compute_rope_with_pos_ids",
    "RoPE",
]

# python libraries
from pathlib import Path
import math

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


def yarn_attention_mscale(factor: float) -> float:
    """YaRN 的注意力温度补偿：推断时对 q、k 各乘 mscale。

    使用 NTK-by-parts 插值后，attention logits 尺度变化，
    需乘以 ``mscale = 0.1·ln(factor) + 1`` 恢复对比度（YaRN §3.3）。
    """
    return 0.1 * math.log(factor) + 1.0


def precompute_rope_params(head_dim, theta_base=10_000, context_length=4096,
                           freq_config=None, dtype=torch.float32,
                           ntk_factor=None, yarn_config=None):
    """预计算 RoPE 的 cos/sin。

    频率调整（互斥，择一）：
    - ``freq_config``：Llama-3 式分段缩放（低频降频 + 中频平滑过渡）
    - ``ntk_factor``：NTK-aware 外推——放大 theta_base
      ``θ' = θ · factor^(d/(d-2))``，等效降频，支持长上下文
    - ``yarn_config``：YaRN NTK-by-parts——按波长分带：
      短波长（高频）不变、长波长（低频）除以 factor、中带按 ramp 混合；
      需配合 ``yarn_attention_mscale(factor)`` 缩放 q/k。

    Returns:
        (cos, sin)，各 (context_length, head_dim)
    """
    assert head_dim % 2 == 0, "Embedding dimension must be even"

    # NTK-aware：先放大 theta_base
    if ntk_factor is not None:
        theta_base = theta_base * (ntk_factor ** (head_dim / (head_dim - 2)))

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

    if yarn_config is not None:
        # YaRN NTK-by-parts（按波长分带）
        factor = yarn_config["factor"]
        orig_ctx = yarn_config["original_context_length"]
        beta_fast = yarn_config.get("beta_fast", 32)    # 高频带边界
        beta_slow = yarn_config.get("beta_slow", 1)     # 低频带边界
        wavelen = 2 * torch.pi / inv_freq
        high_wavelen = orig_ctx / beta_fast             # 短波长：不缩放
        low_wavelen = orig_ctx / beta_slow              # 长波长：完全插值
        ramp = ((orig_ctx / wavelen) - beta_slow) / (beta_fast - beta_slow)
        ramp = ramp.clamp(0.0, 1.0)
        inv_freq = torch.where(
            wavelen < high_wavelen, inv_freq,
            torch.where(
                wavelen > low_wavelen, inv_freq / factor,
                (1 - ramp) * inv_freq / factor + ramp * inv_freq,
            ),
        )

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


class RoPE(nn.Module):
    """模块化 RoPE：预计算 cos/sin 注册为 buffer，forward 时应用旋转。

    与函数式 ``precompute_rope_params`` + ``compute_rope`` 等价，
    便于按 nn.Module 方式组装进模型；``rope_freq`` 透传 Llama-3
    分段频率缩放配置。
    """

    def __init__(self, head_dim, theta_base = 10_000.0, context_length = 4096,
                 rope_freq = None, dtype = torch.float32):
        super().__init__()

        cos, sin = precompute_rope_params(
            head_dim = head_dim,
            theta_base = theta_base,
            context_length = context_length,
            freq_config = rope_freq,
            dtype = dtype,
        )
        self.register_buffer("cos", cos, persistent = False)
        self.register_buffer("sin", sin, persistent = False)

    def forward(self, x, offset = 0):
        return compute_rope(x, self.cos, self.sin, offset = offset)

    def forward_with_pos_ids(self, x, pos_ids):
        return compute_rope_with_pos_ids(x, self.cos, self.sin, pos_ids)




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
