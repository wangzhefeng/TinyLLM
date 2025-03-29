# -*- coding: utf-8 -*-

# ***************************************************
# * File        : embedding.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-01-24
# * Version     : 1.0.012400
# * Description : https://github.com/rasbt/LLMs-from-scratch/blob/2dc46bedc6e86b79a16c4099e557564cd23e03ef/ch02/03_bonus_embedding-vs-matmul/embeddings-and-linear-layers.ipynb
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# * TODO        : 1.
# ***************************************************

__all__ = []

# python libraries
import os
import sys
ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import torch

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


def token_embedding():
    pass


def position_embedding():
    pass


def embedding():
    pass


def precompute_rope_params(head_dim, theta_base=10_000, context_length=4096):
    assert head_dim % 2 == 0, "Embedding dimension must be even"

    # Compute the inverse frequencies
    inv_freq = 1.0 / (theta_base ** (torch.arange(0, head_dim, 2)[: (head_dim // 2)].float() / head_dim))

    # Generate position indices
    positions = torch.arange(context_length)

    # Compute the angles
    angles = positions[:, None] * inv_freq[None, :]  # Shape: (context_length, head_dim // 2)

    # Expand angles to match the head_dim
    angles = torch.cat([angles, angles], dim=1)  # Shape: (context_length, head_dim)

    # Precompute sine and cosine
    cos = torch.cos(angles)
    sin = torch.sin(angles)

    return cos, sin


def compute_rope(x, cos, sin):
    """
    RoPE: RoFormer: Enhanced Transformer with Rotary Position Embedding (2021)
    https://arxiv.org/abs/2104.09864
    """
    # x: (batch_size, num_heads, seq_len, head_dim)
    batch_size, num_heads, seq_len, head_dim = x.shape
    assert head_dim % 2 == 0, "Head dimension must be even"

    # Split x into first half and second half
    x1 = x[..., : head_dim // 2]  # First half
    x2 = x[..., head_dim // 2 :]  # Second half

    # Adjust sin and cos shapes
    cos = cos[:seq_len, :].unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, seq_len, head_dim)
    sin = sin[:seq_len, :].unsqueeze(0).unsqueeze(0)

    # Apply the rotary transformation
    rotated = torch.cat((-x2, x1), dim=-1)
    x_rotated = (x * cos) + (rotated * sin)

    return x_rotated.to(dtype=x.dtype)




# 测试代码 main 函数
def main():
    # Settings
    batch_size = 2
    context_len = 5
    num_heads = 4
    head_dim = 16

    # Instantiate RoPE parameters
    cos, sin = precompute_rope_params(head_dim=head_dim, context_length=context_len)

    # Dummy query and key tensors
    torch.manual_seed(123)
    queries = torch.randn(batch_size, num_heads, context_len, head_dim)
    keys = torch.randn(batch_size, num_heads, context_len, head_dim)

    # Apply rotary position embeddings
    queries_rot = compute_rope(queries, cos, sin)
    keys_rot = compute_rope(keys, cos, sin)

if __name__ == "__main__":
    main()
