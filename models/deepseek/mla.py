# -*- coding: utf-8 -*-

"""Multi-Head Latent Attention (MLA) placeholder.

MLA 最早由 DeepSeek-V2 提出，并被 DeepSeek-V3 / R1 沿用。
核心思想：把 K/V 压缩到低维 latent，推理时只缓存 latent，再反投影出 K/V，
从而显著降低 KV cache 显存。

本文件当前只是骨架，避免主仓库出现“藏在 layers/attention.py 里的空类”。
实现时参考：
- DeepSeek-V2: https://arxiv.org/abs/2405.04434
- DeepSeek-V3: https://arxiv.org/abs/2412.19437
"""

from __future__ import annotations

import torch
from torch import nn


class MultiHeadLatentAttention(nn.Module):
    """Placeholder for MLA.

    TODO:
    - query / key / value 的低秩压缩投影
    - RoPE 只作用于部分维度（decoupled RoPE）
    - KV cache 只保存压缩后的 latent
    - 与 GQA 的对比实验
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        raise NotImplementedError(
            "MLA is not implemented yet. See models/deepseek/README.md for the roadmap."
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
