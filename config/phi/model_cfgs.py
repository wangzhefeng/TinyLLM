# -*- coding: utf-8 -*-

# ***************************************************
# * File        : model_cfgs.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2026-10-24
# * Version     : 1.0.102421
# * Description : Phi-1.5 模型配置（官方尺寸 + 教学缩小版）。
# *               来源：https://huggingface.co/microsoft/phi-1_5 config.json
# * Requirement : torch
# ***************************************************

# python libraries
from pathlib import Path

import torch

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


# 1.3B phi-1_5（官方尺寸）
PHI15_CONFIG = {
    "vocab_size": 51200,          # 含 [PAD]=50257 等扩展 special tokens
    "context_length": 2048,
    "embed_dim": 2048,
    "n_heads": 32,
    "n_layers": 24,
    "hidden_dim": 8192,           # intermediate_size
    "d_ff": 8192,
    "head_dim": 64,
    "partial_rotary_factor": 0.5, # RoPE 只作用每头前 32 维
    "rope_base": 10_000.0,
    "dropout": 0.0,
    "dtype": torch.float32,
}

# 教学缩小版（单测用）
PHI15_TINY_CONFIG = {
    "vocab_size": 1000,
    "context_length": 128,
    "embed_dim": 64,
    "n_heads": 4,
    "n_layers": 2,
    "hidden_dim": 256,
    "d_ff": 256,
    "head_dim": 16,
    "partial_rotary_factor": 0.5,
    "rope_base": 10_000.0,
    "dropout": 0.0,
    "dtype": torch.float32,
}


def get_config(name: str = "phi-1_5"):
    from config.base import ModelConfig

    configs = {
        "phi-1_5": PHI15_CONFIG,
        "tiny": PHI15_TINY_CONFIG,
    }
    key = name.lower()
    if key not in configs:
        raise ValueError(f"Unknown phi config '{name}'; choose from {sorted(configs)}")
    return ModelConfig.from_dict(configs[key])


# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
