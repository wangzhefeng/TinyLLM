# -*- coding: utf-8 -*-

# ***************************************************
# * File        : qwen3_model.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-07-27
# * Version     : 1.1.072816
# * Description : Qwen3Model 主干网络
# *               ├── embed_tokens: VocabParallelEmbedding 词嵌入层
# *               ├── layers: 多个 Qwen3DecoderLayer 解码器层
# *               └── norm: FinalRMSNorm 最终归一化层
# *               输入为展平的 token id 序列(varlen batch)，各序列的边界
# *               信息由 utils.context.Context 传递给 Attention 层。
# * Link        : https://github.com/GeeeekExplorer/nano-vllm
# * Requirement : torch >= 2.0
# ***************************************************

__all__ = [
    "Qwen3Model",
]

# python libraries
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn

from ..layers.decoder import Qwen3DecoderLayer
from ..layers.embed import VocabParallelEmbedding
from ..layers.final_norm import FinalRMSNorm

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


class Qwen3Model(nn.Module):
    """
    主要的模型结构，包括：
    1. VocabParallelEmbedding: 词嵌入层
    2. 多个 Qwen3DecoderLayer: 解码器层
    3. FinalRMSNorm: 最终的归一化层
    """

    def __init__(self, config):
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            Qwen3DecoderLayer(config) for _ in range(config.num_hidden_layers)
        ])
        self.norm = FinalRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            input_ids: 展平的 token id, shape [total_tokens] (prefill) 或 [bs] (decode)
            positions: 每个 token 的位置 id, 与 input_ids 同形
        Returns:
            hidden_states: shape [total_tokens, hidden_size]
        """
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
