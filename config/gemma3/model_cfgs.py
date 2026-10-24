# -*- coding: utf-8 -*-

# ***************************************************
# * File        : model_cfgs.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2026-10-24
# * Version     : 1.0.102421
# * Description : Gemma3 模型配置（教学版 270M，结构对齐 google/gemma-3-270m）。
# * Requirement : torch
# ***************************************************

# python libraries
from pathlib import Path

import torch

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


GEMMA3_CONFIG_270M = {
    "vocab_size": 262_144,
    "context_length": 32_768,
    "embed_dim": 640,
    "n_heads": 4,
    "n_layers": 18,
    "hidden_dim": 2048,
    "d_ff": 2048,
    "head_dim": 256,
    "qk_norm": True,
    "n_kv_groups": 1,
    "rope_local_base": 10_000.0,
    "rope_base": 1_000_000.0,
    "sliding_window": 512,
    "layer_types": [
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "full_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "full_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "full_attention",
    ],
    "dtype": torch.bfloat16,
    "query_pre_attn_scalar": 256,
}


def get_config(name: str = "270m"):
    from config.base import ModelConfig

    configs = {
        "270m": GEMMA3_CONFIG_270M,
    }
    key = name.lower()
    if key not in configs:
        raise ValueError(f"Unknown Gemma3 config '{name}'; choose from {sorted(configs)}")
    return ModelConfig.from_dict(configs[key])


# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
