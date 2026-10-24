# -*- coding: utf-8 -*-

# ***************************************************
# * File        : ALiBi.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2026-10-20
# * Version     : 1.0.102020
# * Description : ALiBi (Attention with Linear Biases) 位置编码。
# *               不给 token 加位置向量，而是在 attention 分数上加一个
# *               与 (query-key 距离) 成正比的线性负偏置，斜率按头分配。
# *               天然支持长度外推（训练短、推断长）。
# * Link        : Train Short, Test Long: https://arxiv.org/abs/2108.12409
# * Requirement : torch
# ***************************************************

__all__ = [
    "get_alibi_slopes",
    "build_alibi_bias",
]

# python libraries
import math
from pathlib import Path

import torch

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


def get_alibi_slopes(n_heads: int) -> torch.Tensor:
    """按头计算 ALiBi 斜率：对 2 的幂次头数取 2^(-8i/n)。

    n_heads=8 时：[1/2, 1/4, ..., 1/256]（论文 §3.3）。
    非二次幂头数：对最近的二次幂序列做插值（论文附录 B）。

    Returns:
        (n_heads,) float32
    """
    def power_of_two_slopes(n):
        # 论文 §3.3：slopes = 2^(-8i/n), i = 1..n（n 为 2 的幂）
        return torch.tensor([2 ** (-8 * (i + 1) / n) for i in range(n)])

    if math.log2(n_heads).is_integer():
        return power_of_two_slopes(n_heads)

    # 非二次幂：取 2·closest 序列的偶数下标做几何插值（论文附录 B 的标准实现）
    closest = 2 ** math.floor(math.log2(n_heads))
    slopes = power_of_two_slopes(closest)
    extra = power_of_two_slopes(2 * closest)[0::2][: n_heads - closest]
    return torch.cat([slopes, extra]).float()


def build_alibi_bias(n_heads: int, seq_len: int, device=None,
                     causal: bool = True, dtype=None) -> torch.Tensor:
    """构造可加到 attention 分数上的 ALiBi 偏置。

    ``bias[h, i, j] = -slope_h * (i - j)``（j<=i，距离越远惩罚越大）；
    causal=True 时未来位置（j>i）置 -inf。

    可直接传给 ``scaled_dot_product_attention(attn_mask=bias)``。

    Args:
        dtype: 输出 dtype；None（默认）保持 float32，内部始终以 float32 计算后再转换。

    Returns:
        (1, n_heads, seq_len, seq_len)，dtype 为 dtype 或 float32
    """
    slopes = get_alibi_slopes(n_heads).to(device)                      # (H,)
    pos = torch.arange(seq_len, device=device, dtype=torch.float32)
    dist = pos.view(-1, 1) - pos.view(1, -1)                           # i - j
    bias = -slopes.view(1, -1, 1, 1) * dist.view(1, 1, seq_len, seq_len)
    if causal:
        future = torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1
        )
        bias = bias.masked_fill(future.view(1, 1, seq_len, seq_len), float("-inf"))
    if dtype is not None:
        bias = bias.to(dtype)
    return bias


# 测试代码 main 函数
def main():
    slopes = get_alibi_slopes(8)
    print(f"ALiBi slopes (n_heads=8): {slopes.tolist()}")
    bias = build_alibi_bias(n_heads=8, seq_len=6)
    print(f"bias shape: {tuple(bias.shape)}")

if __name__ == "__main__":
    main()
