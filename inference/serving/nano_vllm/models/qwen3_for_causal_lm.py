# -*- coding: utf-8 -*-

# ***************************************************
# * File        : qwen3_for_causal_lm.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-07-27
# * Version     : 1.1.072816
# * Description : Qwen3ForCausalLM 因果语言模型
# *               ├── model: Qwen3Model 主干
# *               ├── lm_head: ParallelLMHead 输出层
# *               ├── forward(): 返回最后一层 hidden_states
# *               └── compute_logits(): hidden_states -> vocab logits
# *               注意：prefill 时只对每条序列最后一个位置的 hidden_states
# *               计算 logits(见 ModelRunner.run_model)。
# * Link        : https://github.com/GeeeekExplorer/nano-vllm
# * Requirement : torch >= 2.0
# ***************************************************

__all__ = [
    "Qwen3ForCausalLM",
]

# python libraries
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)
PKG_ROOT = str(Path(__file__).resolve().parents[1])
if PKG_ROOT in sys.path:
    sys.path.remove(PKG_ROOT)
sys.path.insert(0, PKG_ROOT)
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn

from models.parallel_lm_head import ParallelLMHead
from models.qwen3_model import Qwen3Model

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


class Qwen3ForCausalLM(nn.Module):
    """
    因果语言模型，其主要组成部分包括：
        1. Qwen3Model: 主要的模型结构
        2. ParallelLMHead: 语言模型头，用于生成最终的输出 logits
    """
    # 拆分权重 -> 打包参数的映射(供 utils.loader.load_state_dict 使用)
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = Qwen3Model(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(input_ids, positions)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.lm_head(hidden_states)




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
