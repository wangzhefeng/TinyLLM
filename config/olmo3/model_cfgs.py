# -*- coding: utf-8 -*-

# ***************************************************
# * File        : model_cfgs.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2026-10-24
# * Version     : 1.0.102401
# * Description : Olmo 3 配置（allenai, 2025）。
# *               架构结构已对照 transformers 官方源码核实：
# *               models/olmo3/{configuration,modular}_olmo3.py 与
# *               models/olmo2/modeling_olmo2.py（post-norm block、flat 级非参数
# *               QK-norm、无 bias SwiGLU、RoPE theta=500000、3:1 滑窗/full 混合层）。
# *               注意：allenai/OLMo-3* 为 gated repo（HTTP 401），各尺寸
# *               checkpoint 级 config.json 未能直接核实，尺寸超参以
# *               transformers Olmo3Config 类默认值为准（context_length 除外，
# *               见条目内注释）。
# * Link        : https://huggingface.co/allenai/OLMo-3-7B-Instruct
# *               https://github.com/huggingface/transformers/tree/main/src/transformers/models/olmo3
# * Requirement : torch
# ***************************************************

# python libraries
from pathlib import Path

import torch

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


# tiny 测试配置：4 层（3 层 sliding + 第 4 层 full），窗口 6 < context 32
OLMO3_TINY_CONFIG = {
    "vocab_size": 64,               # Vocabulary size
    "context_length": 32,           # Context length for RoPE precompute
    "embed_dim": 32,                # Embedding dimension
    "n_heads": 4,                   # Number of attention heads
    "n_layers": 4,                  # Number of layers
    "d_ff": 64,                     # Size of the intermediate dimension in FeedForward
    "head_dim": 8,                  # Size of the heads in GQA
    "qk_norm": True,                # Olmo 3 恒为 True（非参数化，见 olmo3.py）
    "n_kv_groups": 2,               # Key-Value groups for grouped-query attention
    "rope_base": 500_000.0,         # The base in RoPE's "theta"
    "dtype": torch.float32,
    "num_experts": 0,
    "dropout": 0.0,
    "sliding_window": 6,            # 滑动窗口大小（sliding_attention 层）
    "full_attention_interval": 4,   # 每 4 层 1 层 full attention，其余滑窗
    "tie_word_embeddings": True,    # 测试权重共享路径
}


def get_cfgs(CHOOSE_MODEL = "7B"):
    if CHOOSE_MODEL == "7B":
        # transformers Olmo3Config 类默认值（官方代码）：
        # vocab 50304 / hidden 4096 / inter 11008 / 32 层 / 32 头 /
        # num_key_value_heads 缺省 = num_attention_heads（MHA）/ eps 1e-5 /
        # rope_theta 500000 / sliding_window 4096
        # context_length 取模型卡公开的 65,536（类默认 max_position_embeddings=2048
        # 为 OLMo-2 遗留值，且小于滑动窗口，教学上取模型卡值更合理）
        OLMO3_CONFIG = {
            "vocab_size": 50_304,           # Vocabulary size
            "context_length": 65_536,       # Context length (模型卡公开值)
            "embed_dim": 4_096,             # Embedding dimension
            "n_heads": 32,                  # Number of attention heads
            "n_layers": 32,                 # Number of layers
            "d_ff": 11_008,                 # Size of the intermediate dimension in FeedForward
            "head_dim": 128,                # Size of the heads in GQA
            "qk_norm": True,                # Olmo 3 恒为 True（非参数化，见 olmo3.py）
            "n_kv_groups": 32,              # 类默认 num_key_value_heads = num_attention_heads（MHA）
            "rope_base": 500_000.0,         # The base in RoPE's "theta"
            "dtype": torch.bfloat16,        # Lower-precision dtype to reduce memory usage
            "num_experts": 0,
            "sliding_window": 4_096,        # 滑动窗口大小（sliding_attention 层）
            "full_attention_interval": 4,   # 每 4 层 1 层 full attention，其余滑窗
            "tie_word_embeddings": False,
        }
    elif CHOOSE_MODEL == "TINY":
        OLMO3_CONFIG = dict(OLMO3_TINY_CONFIG)
    else:
        raise ValueError(f"{CHOOSE_MODEL} is not supported.")

    from config.base import ModelConfig

    return ModelConfig.from_dict(OLMO3_CONFIG)





# 测试代码 main 函数
def main():
    OLMO3_CONFIG = get_cfgs()
    print(OLMO3_CONFIG)
    print(get_cfgs("TINY"))

if __name__ == "__main__":
    main()
