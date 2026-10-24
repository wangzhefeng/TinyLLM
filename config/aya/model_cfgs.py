# -*- coding: utf-8 -*-

# ***************************************************
# * File        : model_cfgs.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2026-10-24
# * Version     : 1.0.102421
# * Description : Tiny Aya（Cohere，260M）模型配置。
# *               ⚠ 官方 config.json 未经核实（cohere-for-ai/tinyaya 为
# *               门控模型，HTTP 401）：AYA_CONFIG_260M 为按 260M 参数量级
# *               推算的 Command 家族风格假设配置（GQA + CAT + uvqk +
# *               weight tying + SwiGLU，粗算约 2.7 亿参数，其中 tied 词嵌入
# *               约 1.97 亿）；AYA_TINY_CONFIG 为教学用缩小配置。
# * Requirement : torch
# ***************************************************

# python libraries
from pathlib import Path

import torch

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


# 假设配置（未经官方核实）：词表沿用 Cohere 家族 256k 多语词表，
# tied embedding 256000 x 768 ≈ 197M，12 层 x ~6.3M/层 ≈ 76M，合计 ≈ 272M
AYA_CONFIG_260M = {
    "vocab_size": 256_000,
    "context_length": 8_192,
    "embed_dim": 768,
    "n_heads": 12,
    "n_kv_groups": 4,                   # GQA: 12 Q heads : 4 KV heads
    "head_dim": 64,
    "n_layers": 12,
    "hidden_dim": 2_048,
    "d_ff": 2_048,                      # SwiGLU 中间维
    "rope_base": 10_000.0,              # CAT 旋转频率基（theta_base）
    "dtype": torch.float32,
}


# 教学用 TINY 配置（离线 CPU 测试）
AYA_TINY_CONFIG = {
    "vocab_size": 128,
    "context_length": 32,
    "embed_dim": 32,
    "n_heads": 4,
    "n_kv_groups": 2,
    "head_dim": 8,
    "n_layers": 2,
    "hidden_dim": 64,
    "d_ff": 64,
    "rope_base": 10_000.0,
    "dtype": torch.float32,
}


def get_config(name: str = "260m"):
    configs = {
        "260m": AYA_CONFIG_260M,
        "tiny": AYA_TINY_CONFIG,
    }
    key = name.lower()
    if key not in configs:
        raise ValueError(f"Unknown Aya config '{name}'; choose from {sorted(configs)}")
    # 注意：ModelConfig dataclass 会丢弃 GQA/CAT 等教学专用键，
    # 因此返回保留全部键的 DotDict（属性访问与 ModelConfig 一致）
    from utils.args_tools import DotDict

    return DotDict(configs[key])


# 测试代码 main 函数
def main():
    pass


if __name__ == "__main__":
    main()
