# -*- coding: utf-8 -*-

# ***************************************************
# * File        : model_cfg.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-10-02
# * Version     : 1.0.100201
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
from pathlib import Path

import torch

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


# 7B Llama2
LLAMA2_CONFIG_7B = {
    "vocab_size": 32000,     # Vocabulary size
    "context_length": 4096,  # Context length
    "embed_dim": 4096,       # Embedding dimension
    "n_heads": 32,           # Number of attention heads
    "n_layers": 32,          # Number of layers
    "hidden_dim": 11008,     # NEW: Size of the intermediate dimension in FeedForward
    "dtype": torch.bfloat16  # NEW: Lower-precision dtype to reduce memory usage
}


# 8B Llama3
LLAMA3_CONFIG_8B = {
    "vocab_size": 128_256,   # NEW: Larger vocabulary size
    "context_length": 8192,  # NEW: Larger context length
    "embed_dim": 4096,       # Embedding dimension
    "n_heads": 32,           # Number of attention heads
    "n_layers": 32,          # Number of layers
    "hidden_dim": 14_336,    # NEW: Larger size of the intermediate dimension in FeedForward
    "n_kv_groups": 8,        # NEW: Key-Value groups for grouped-query attention
    "rope_base": 500_000.0,  # NEW: The base in RoPE's "theta" was increased to 500_000
    "rope_freq": None,       # NEW: Additional configuration for adjusting the RoPE frequencies
    "dtype": torch.bfloat16  # Lower-precision dtype to reduce memory usage
}

# 8B Llama3.1
LLAMA31_CONFIG_8B = {
    "vocab_size": 128_256,      # Vocabulary size
    "context_length": 131_072,  # NEW: Larger supported context length
    "embed_dim": 4096,            # Embedding dimension
    "n_heads": 32,              # Number of attention heads
    "n_layers": 32,             # Number of layers
    "hidden_dim": 14_336,       # Size of the intermediate dimension in FeedForward
    "n_kv_groups": 8,           # Key-Value groups for grouped-query attention
    "rope_base": 500_000.0,     # The base in RoPE's "theta"
    "dtype": torch.bfloat16,    # Lower-precision dtype to reduce memory usage
    "rope_freq": {              # NEW: RoPE frequency scaling
        "factor": 8.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_context_length": 8192,
    }
}

# 1B Llama3.2
LLAMA32_CONFIG_1B = {
    "vocab_size": 128_256,      # Vocabulary size
    "context_length": 131_072,  # Context length
    "embed_dim": 2048,          # Embedding dimension
    "n_heads": 32,              # Number of attention heads
    "n_layers": 16,             # Number of layers
    "hidden_dim": 8192,         # Size of the intermediate dimension in FeedForward
    "n_kv_groups": 8,           # Key-Value groups for grouped-query attention
    "rope_base": 500_000.0,     # The base in RoPE's "theta"
    "dtype": torch.bfloat16,    # Lower-precision dtype to reduce memory usage
    "rope_freq": {              # RoPE frequency scaling
        "factor": 32.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_context_length": 8192,
    },
    "tie_word_embeddings": True,   # NEW: 官方 1B 权重 out_head 与 tok_embed 共享
}




def get_config(name="llama3_8b"):
    from config.base import ModelConfig

    configs = {
        "llama2": LLAMA2_CONFIG_7B,
        "llama3_8b": LLAMA3_CONFIG_8B,
        "llama31_8b": LLAMA31_CONFIG_8B,
        "llama32_1b": LLAMA32_CONFIG_1B,
    }
    key = name.lower()
    if key not in configs:
        raise ValueError(f"Unknown Llama config '{name}'; choose from {sorted(configs)}")
    return ModelConfig.from_dict(configs[key])


# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
