# -*- coding: utf-8 -*-

# ***************************************************
# * File        : qwen_model_config.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-08-31
# * Version     : 1.0.083123
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
from pathlib import Path

import torch

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


def get_cfgs(CHOOSE_MODEL = "0.6B"):
    if CHOOSE_MODEL == "0.6B":
        QWEN3_CONFIG = {
            "vocab_size": 151_936,           # Vocabulary size
            "context_length": 40_960,        # Context length that was used to train the model
            "embed_dim": 1024,               # Embedding dimension
            "n_heads": 16,                   # Number of attention heads
            "n_layers": 28,                  # Number of layers
            "d_ff": 3072,                    # Size of the intermediate dimension in FeedForward
            "head_dim": 128,                 # Size of the heads in GQA
            "qk_norm": True,                 # Whether to normalize queries and keys in GQA
            "n_kv_groups": 8,                # Key-Value groups for grouped-query attention
            "rope_base": 1_000_000.0,        # The base in RoPE's "theta"
            "dtype": torch.bfloat16,         # Lower-precision dtype to reduce memory usage
            "num_experts": 0,
        }
    elif CHOOSE_MODEL == "1.7B":
        QWEN3_CONFIG = {
            "vocab_size": 151_936,
            "context_length": 40_960,
            "embed_dim": 2048,               # 2x larger than above
            "n_heads": 16,
            "n_layers": 28,
            "d_ff": 6144,                    # 2x larger than above
            "head_dim": 128,
            "qk_norm": True,
            "n_kv_groups": 8,
            "rope_base": 1_000_000.0,
            "dtype": torch.bfloat16,
            "num_experts": 0,
        }
    elif CHOOSE_MODEL == "4B":
        QWEN3_CONFIG = {
            "vocab_size": 151_936,
            "context_length": 40_960,
            "embed_dim": 2560,               # 25% larger than above
            "n_heads": 32,                   # 2x larger than above
            "n_layers": 36,                  # 29% larger than above
            "d_ff": 9728,                    # ~3x larger than above
            "head_dim": 128,
            "qk_norm": True,
            "n_kv_groups": 8,
            "rope_base": 1_000_000.0,
            "dtype": torch.bfloat16,
            "num_experts": 0,
        }  
    elif CHOOSE_MODEL == "8B":
        QWEN3_CONFIG = {
            "vocab_size": 151_936,
            "context_length": 40_960,
            "embed_dim": 4096,               # 60% larger than above
            "n_heads": 32,
            "n_layers": 36,                  # 26% larger than above
            "d_ff": 12288,
            "head_dim": 128,
            "qk_norm": True,
            "n_kv_groups": 8,
            "rope_base": 1_000_000.0,
            "dtype": torch.bfloat16,
            "num_experts": 0,
        } 
    elif CHOOSE_MODEL == "14B":
        QWEN3_CONFIG = {
            "vocab_size": 151_936,
            "context_length": 40_960,
            "embed_dim": 5120,               # 25% larger than above
            "n_heads": 40,                   # 25% larger than above
            "n_layers": 40,                  # 11% larger than above
            "d_ff": 17408,                   # 42% larger than above
            "head_dim": 128,
            "qk_norm": True,
            "n_kv_groups": 8,
            "rope_base": 1_000_000.0,
            "dtype": torch.bfloat16,
            "num_experts": 0,
        } 
    elif CHOOSE_MODEL == "32B":
        QWEN3_CONFIG = {
            "vocab_size": 151_936,
            "context_length": 40_960,
            "embed_dim": 5120,                
            "n_heads": 64,                   # 60% larger than above
            "n_layers": 64,                  # 60% larger than above
            "d_ff": 25600,                   # 47% larger than above
            "head_dim": 128,
            "qk_norm": True,
            "n_kv_groups": 8,
            "rope_base": 1_000_000.0,
            "dtype": torch.bfloat16,
            "num_experts": 0,
        }
    elif CHOOSE_MODEL == "30B-A3B":
        # Same config for
        # https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct (Qwen3 Coder Flash)
        # https://huggingface.co/Qwen/Qwen3-30B-A3B-Thinking-2507
        # https://huggingface.co/Qwen/Qwen3-235B-A22B-Instruct-2507
        # https://huggingface.co/Qwen/Qwen3-30B-A3B (original Instruct/Thinking hybrid model)
        QWEN3_CONFIG = {
            "vocab_size": 151_936,
            "context_length": 262_144,
            "embed_dim": 2048,                
            "n_heads": 32,
            "n_layers": 48,
            # MoE 模型无 d_ff，FFN 维度见 moe_intermediate_size
            "head_dim": 128,
            "qk_norm": True,
            "n_kv_groups": 4,
            "rope_base": 10_000_000.0,
            "dtype": torch.bfloat16,
            "num_experts": 128,
            "num_experts_per_tok": 8,
            "moe_intermediate_size": 768,
        }
    elif CHOOSE_MODEL == "Qwen3.5-4B":
        # 官方 config.json 已核实：https://huggingface.co/Qwen/Qwen3.5-4B（text_config）
        # 多模态视觉塔与 MTP（多 token 预测）头不在本教学实现范围内
        # （models/qwen3/qwen3_5.py 仅实现文本栈）
        # 结构：3:1 混合层（gated delta net 线性注意力 + full attention），
        # full 层带 sigmoid 输出门控与 partial RoPE(25%)，权重共享 output head
        QWEN3_CONFIG = {
            "vocab_size": 248_320,
            "context_length": 262_144,
            "embed_dim": 2_560,
            "n_heads": 16,
            "n_layers": 32,
            "d_ff": 9_216,
            "head_dim": 256,
            "qk_norm": True,                 # full 层逐 head QK-norm（恒开启）
            "n_kv_groups": 4,
            "rope_base": 10_000_000.0,
            "dtype": torch.bfloat16,
            "num_experts": 0,
            "full_attention_interval": 4,    # 每 4 层 1 层 full attention，其余 linear attention
            "attn_output_gate": True,        # full 层 q_proj 输出二分为 query+gate，sigmoid 门控
            "partial_rotary_factor": 0.25,   # 仅前 25% head 维度参与 RoPE
            "linear_num_key_heads": 16,      # 线性注意力 key 头数
            "linear_key_head_dim": 128,      # 线性注意力 key 头维
            "linear_num_value_heads": 32,    # 线性注意力 value 头数
            "linear_value_head_dim": 128,    # 线性注意力 value 头维
            "linear_conv_kernel_dim": 4,     # 线性注意力因果 depthwise 卷积核
            "tie_word_embeddings": True,
        }
    else:
        raise ValueError(f"{CHOOSE_MODEL} is not supported.")

    from config.base import ModelConfig

    return ModelConfig.from_dict(QWEN3_CONFIG)




# 测试代码 main 函数
def main():
    QWEN3_CONFIG = get_cfgs()
    print(QWEN3_CONFIG)

if __name__ == "__main__":
    main()
