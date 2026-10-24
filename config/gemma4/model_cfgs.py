# -*- coding: utf-8 -*-

# ***************************************************
# * File        : model_cfgs.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2026-10-24
# * Version     : 1.0.102421
# * Description : Gemma 4 模型配置。E2B/E4B 参数于 2026-10-24 逐项核对
# *               google/gemma-4-e2b、google/gemma-4-e4b 的官方 config.json
# *               （text_config 部分）；TINY 为教学用缩小配置。
# *
# *               官方 config 中存在、教学模型暂未消费的键（备查）：
# *               num_kv_shared_layers（末 20/18 层跨层 KV 共享）、
# *               use_double_wide_mlp（仅 KV 共享层 FFN 加宽一倍，E4B 为
# *               False）、hidden_size_per_layer_input / vocab_size_per_layer_input
# *               （PLE per-layer embeddings）、attention_k_eq_v、enable_moe_block。
# * Requirement : torch
# ***************************************************

# python libraries
from pathlib import Path

import torch

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


# google/gemma-4-e2b text_config（官方核实）
# n_layers=35，local:global = 4:1（full 落在 4, 9, ..., 34，末层 full）
GEMMA4_CONFIG_E2B = {
    "vocab_size": 262_144,
    "context_length": 131_072,          # max_position_embeddings
    "embed_dim": 1_536,                 # hidden_size
    "n_heads": 8,
    "n_layers": 35,
    "hidden_dim": 6_144,
    "d_ff": 6_144,                      # intermediate_size
    "head_dim": 256,                    # sliding 层 head_dim
    "global_head_dim": 512,             # full 层 head_dim（官方 per-layer 覆盖）
    "qk_norm": True,
    "n_kv_groups": 1,                   # num_key_value_heads
    "rope_local_base": 10_000.0,        # sliding：default 全旋转
    "rope_base": 1_000_000.0,           # full：proportional partial RoPE
    "partial_rotary_factor": 0.25,      # full 层旋转 25% 维度
    "sliding_window": 512,
    "local_per_global": 4,              # 4:1 交错比
    "final_logit_softcapping": 30.0,
    "query_pre_attn_scalar": 1,         # 官方 attention scaling = 1.0
    # ---- 以下为官方存在、教学版未实现的效率特性（备查） ----
    "num_kv_shared_layers": 20,
    "use_double_wide_mlp": True,
    "hidden_size_per_layer_input": 256,
    "vocab_size_per_layer_input": 262_144,
    "dtype": torch.bfloat16,
}


# google/gemma-4-e4b text_config（官方核实）
# n_layers=42，local:global = 5:1（full 落在 5, 11, ..., 41，末层 full）
GEMMA4_CONFIG_E4B = {
    "vocab_size": 262_144,
    "context_length": 131_072,
    "embed_dim": 2_560,
    "n_heads": 8,
    "n_layers": 42,
    "hidden_dim": 10_240,
    "d_ff": 10_240,
    "head_dim": 256,
    "global_head_dim": 512,
    "qk_norm": True,
    "n_kv_groups": 2,
    "rope_local_base": 10_000.0,
    "rope_base": 1_000_000.0,
    "partial_rotary_factor": 0.25,
    "sliding_window": 512,
    "local_per_global": 5,              # 5:1 交错比
    "final_logit_softcapping": 30.0,
    "query_pre_attn_scalar": 1,
    # ---- 以下为官方存在、教学版未实现的效率特性（备查） ----
    "num_kv_shared_layers": 18,
    "use_double_wide_mlp": False,
    "hidden_size_per_layer_input": 256,
    "vocab_size_per_layer_input": 262_144,
    "dtype": torch.bfloat16,
}


# 教学用 TINY 配置（离线 CPU 测试）：6 层 2:1 交错，2 个 full 层
GEMMA4_TINY_CONFIG = {
    "vocab_size": 256,
    "context_length": 64,
    "embed_dim": 64,
    "n_heads": 4,
    "n_layers": 6,
    "hidden_dim": 128,
    "d_ff": 128,
    "head_dim": 16,                     # sliding 层
    "global_head_dim": 32,              # full 层
    "qk_norm": True,
    "n_kv_groups": 1,
    "rope_local_base": 10_000.0,
    "rope_base": 1_000_000.0,
    "partial_rotary_factor": 0.25,
    "sliding_window": 8,
    "local_per_global": 2,              # 2:1 交错
    "final_logit_softcapping": 30.0,
    "query_pre_attn_scalar": 1,
    "dtype": torch.float32,
}


def get_config(name: str = "e2b"):
    configs = {
        "e2b": GEMMA4_CONFIG_E2B,
        "e4b": GEMMA4_CONFIG_E4B,
        "tiny": GEMMA4_TINY_CONFIG,
    }
    key = name.lower()
    if key not in configs:
        raise ValueError(f"Unknown Gemma4 config '{name}'; choose from {sorted(configs)}")
    # 注意：ModelConfig dataclass 会丢弃 layer 交错/RoPE 等教学专用键，
    # 因此返回保留全部键的 DotDict（属性访问与 ModelConfig 一致）
    from utils.args_tools import DotDict

    return DotDict(configs[key])


# 测试代码 main 函数
def main():
    pass


if __name__ == "__main__":
    main()
