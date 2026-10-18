# -*- coding: utf-8 -*-

# ***************************************************
# * File        : model.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Description : DeepSeek model placeholder
# * Link        : https://github.com/deepseek-ai/DeepSeek-V3
# ***************************************************

"""DeepSeek model assembly (placeholder).

DeepSeek V3 / R1 的完整实现需要：
- Multi-Head Latent Attention (MLA)
- DeepSeekMoE（shared experts + routed experts）
- Multi-Token Prediction (MTP)
- RMSNorm / RoPE / SwiGLU

当前目录只提供骨架与学习笔记，具体实现见 README 的 roadmap。
"""

from __future__ import annotations

import torch
from torch import nn

from models.deepseek.mla import MultiHeadLatentAttention  # noqa: F401  (re-export for now)


class DeepSeekModel(nn.Module):
    """Placeholder for the DeepSeek decoder-only model."""

    def __init__(self, cfg=None):
        super().__init__()
        self.cfg = cfg

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            "DeepSeekModel is not implemented yet. "
            "Start with models/deepseek/mla.py and layers/moe.py."
        )
