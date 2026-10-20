# -*- coding: utf-8 -*-

# ***************************************************
# * File        : rotary_embedding.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-07-28
# * Version     : 1.0.072816
# * Description : 旋转位置编码(RoPE)，纯 PyTorch 实现。
# *               采用 Qwen3 的 "half-split" 旋转方式：
# *               x = [x1, x2] -> x * cos + [-x2, x1] * sin
# * Link        : https://github.com/GeeeekExplorer/nano-vllm
# * Requirement : torch >= 2.0
# ***************************************************

__all__ = [
    "RotaryEmbedding",
    "get_rope",
]

# python libraries
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")
from functools import lru_cache

import torch
import torch.nn as nn

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


class RotaryEmbedding(nn.Module):
    """
    旋转位置编码(RoPE)

    Args:
        head_size: 每个注意力头的维度
        rotary_dim: 应用旋转的维度(通常等于 head_size)
        max_position_embeddings: 最大位置数(决定 cos/sin 缓存大小)
        base: 频率基底(rope_theta)
    """

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
    ):
        super().__init__()
        self.head_size = head_size
        assert rotary_dim == head_size, "教学实现要求 rotary_dim == head_size"
        # inv_freq[i] = base^(-2i / rotary_dim), shape [rotary_dim / 2]
        inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim))
        # t: 位置下标, shape [max_position]
        t = torch.arange(max_position_embeddings, dtype=torch.float32)
        # freqs[pos, i] = pos * inv_freq[i], shape [max_position, rotary_dim / 2]
        freqs = torch.outer(t, inv_freq)
        # half-split 旋转需要把 cos/sin 拼接为完整维度: [max_position, rotary_dim]
        cos = freqs.cos().repeat_interleave(2, dim=-1)
        sin = freqs.sin().repeat_interleave(2, dim=-1)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        """
        [x1, x2] -> [-x2, x1]，x1/x2 为最后一维的前后两半
        """
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            positions: 每个 token 的位置 id, shape [num_tokens]
            query: shape [num_tokens, num_heads, head_dim]
            key:   shape [num_tokens, num_kv_heads, head_dim]
        """
        cos = self.cos[positions].unsqueeze(1)  # [num_tokens, 1, head_dim]
        sin = self.sin[positions].unsqueeze(1)
        query = query * cos + self._rotate_half(query) * sin
        key = key * cos + self._rotate_half(key) * sin
        return query, key


@lru_cache(maxsize=8)
def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
) -> RotaryEmbedding:
    """
    带缓存的 RotaryEmbedding 构造器(同参数共享同一份 cos/sin 缓存)
    """
    return RotaryEmbedding(head_size, rotary_dim, max_position, base)




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
