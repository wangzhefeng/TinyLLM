# -*- coding: utf-8 -*-

# ***************************************************
# * File        : masking.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-07-13
# * Version     : 1.1.102021
# * Description : 注意力掩码工具。统一约定：**True = 屏蔽**（与
# *               layers/attention.py 的 masked_fill 语义一致），
# *               可直接传给 attn_scores.masked_fill(mask, -inf)。
# * Link        : Attention Is All You Need: https://arxiv.org/abs/1706.03762
# * Requirement : torch
# ***************************************************

__all__ = [
    "create_padding_mask",
    "create_look_ahead_mask",
    "subsequent_mask",
    "create_sliding_window_mask",
    "combine_causal_padding_mask",
]

# python libraries
from pathlib import Path

import torch

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


def create_padding_mask(seq, pad_idx=0):
    """Padding 掩码：pad 位置为 True（屏蔽），非 pad 为 False。

    Args:
        seq: (batch_size, seq_len) token ids
        pad_idx: pad token id

    Returns:
        (batch_size, 1, 1, seq_len) bool，True = 屏蔽
    """
    return (seq == pad_idx).unsqueeze(1).unsqueeze(2)


def create_look_ahead_mask(size):
    """因果（look-ahead）掩码：未来位置为 True（屏蔽）。

    Args:
        size: 序列长度

    Returns:
        (size, size) bool，True = 屏蔽。mask[i][j] = j > i
    """
    return torch.triu(torch.ones(size, size, dtype=torch.bool), diagonal=1)


def subsequent_mask(size):
    """create_look_ahead_mask 的别名（保持 True = 屏蔽 约定）。"""
    return create_look_ahead_mask(size)


def create_sliding_window_mask(seq_len, window_size):
    """滑动窗口（SWA）掩码：True = 屏蔽。

    mask[i][j] = True 当且仅当 j > i（未来）或 i - j >= window_size（太远）。

    Args:
        seq_len: 序列长度
        window_size: 每个 token 最多回看的窗口大小（含自身）

    Returns:
        (seq_len, seq_len) bool
    """
    pos = torch.arange(seq_len)
    dist = pos.view(-1, 1) - pos.view(1, -1)                 # i - j
    return (dist < 0) | (dist >= window_size)


def combine_causal_padding_mask(seq, pad_idx=0):
    """组合因果掩码与 padding 掩码（batch 场景）。

    Args:
        seq: (batch_size, seq_len) token ids
        pad_idx: pad token id

    Returns:
        (batch_size, 1, seq_len, seq_len) bool，True = 屏蔽。
        mask[b, 0, i, j] = (j > i) | (token_j 是 pad)
    """
    seq_len = seq.shape[-1]
    causal = create_look_ahead_mask(seq_len)                 # (L, L)
    padding = create_padding_mask(seq, pad_idx=pad_idx)      # (B, 1, 1, L)
    return causal.unsqueeze(0) | padding


# ------------------------------
# test
# ------------------------------
def example_mask():
    """打印 6x6 的因果掩码（True=屏蔽，用 1 显示）。"""
    mask = create_look_ahead_mask(6)
    print(mask.int())


# 测试代码 main 函数
def main():
    example_mask()

if __name__ == "__main__":
    main()
