# -*- coding: utf-8 -*-

# ***************************************************
# * File        : decoder.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-07-27
# * Version     : 1.1.072816
# * Description : Qwen3DecoderLayer 解码器层
# *               ├── input_layernorm: RMSNorm (attn 前, 残差融合)
# *               ├── self_attn: Qwen3Attention
# *               ├── post_attention_layernorm: RMSNorm (mlp 前, 残差融合)
# *               └── mlp: Qwen3MLP
# * Link        : https://github.com/GeeeekExplorer/nano-vllm
# * Requirement : torch >= 2.0
# ***************************************************

__all__ = [
    "Qwen3DecoderLayer",
]

# python libraries
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn

from .attention import Qwen3Attention
from .mlp import Qwen3MLP
from .norm import RMSNorm

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


class Qwen3DecoderLayer(nn.Module):
    """
    解码器层，包括：
    1. Qwen3Attention: 自注意力机制
    2. Qwen3MLP: 前馈神经网络
    3. 两个 RMSNorm: 分别用于注意力前后的层归一化(残差融合形式)
    """

    def __init__(self, config):
        super().__init__()
        self.self_attn = Qwen3Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=config.attention_bias,
            head_dim=config.head_dim,
            rope_theta=config.rope_theta,
        )
        self.mlp = Qwen3MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            # 第一层：残差即输入本身
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
