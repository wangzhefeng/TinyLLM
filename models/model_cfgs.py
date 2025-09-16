# -*- coding: utf-8 -*-

# ***************************************************
# * File        : model_cfgs.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-04-05
# * Version     : 1.0.040500
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

__all__ = [
    "GPT2_124M_CONFIG",
    "GPT_CONFIG_1558M",
    "LLAMA2_CONFIG_7B",
    "LLAMA3_CONFIG_8B",
]

# python libraries
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)

import torch
import tiktoken

from utils.args_tools import DotDict
from utils.device import device_setting

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


# ##############################
# device
# ##############################
device = device_setting()


# ##############################
# model random seed
# ##############################
torch.manual_seed(123)


# ##############################
# GPT2 
# ##############################
# huggingface allowed model names
gpt2_model_names = {
    "gpt2-small (124M)": "gpt2",         # works ok
    "gpt2-medium (355M)": "gpt2-medium", # this file seems to have issues via `generate`
    "gpt2-large (774M)": "gpt2-large",   # works ok
    "gpt2-xl (1558M)": "gpt2-xl"         # works ok
}
gpt2_model_configs = {
    "gpt2-small (124M)": {"embed_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"embed_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"embed_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"embed_dim": 1600, "n_layers": 48, "n_heads": 25},
}
# huggingface gpt2 model
gpt2_huggingface_models = {
    "gpt2-small (124M)": "openai-community/gpt2",
    "gpt2-medium (355M)": "openai-community/gpt2-medium",
    "gpt2-large (774M)": "openai-community/gpt2-large",
    "gpt2-xl (1558M)": "openai-community/gpt2-xl"
}


GPT2_124M_CONFIG = {
    "vocab_size": 50257,     # Vocabular size
    "context_length": 1024,  # Context length
    "max_new_toknes": 200,   # Maximum new tokens to generate
    "embed_dim": 768,        # Embedding dimension
    "d_ff": 4 * 768,         # Hidden dimension
    "n_heads": 12,           # Number of attention heads
    "n_layers": 12,          # Number of transformer layers
    "dropout": 0.1,          # Dropout rate
    "qkv_bias": False,       # Query-Key-Value bias
    "dtype": torch.float32,
    "kv_window_size": 1024,  # KV cache window size
    "top_k": 1,
    "temperature": 0.0,
}
GPT2_124M_CONFIG= DotDict(GPT2_124M_CONFIG)


GPT2_MEDIUM_CONFIG = {
    "vocab_size": 50257,     # Vocabular size
    "context_length": 1024,  # Context length
    "max_new_toknes": 10,    # Maximum new tokens to generate
    "embed_dim": 1024,       # Embedding dimension
    "d_ff": 4 * 1024,        # Hidden dimension
    "n_heads": 16,           # Number of attention heads
    "n_layers": 24,          # Number of transformer layers
    "dropout": 0.1,          # Dropout rate
    "qkv_bias": False,       # Query-Key-Value bias
    "dtype": torch.float32,
    "kv_window_size": 1024,  # KV cache window size
}
GPT2_MEDIUM_CONFIG= DotDict(GPT2_MEDIUM_CONFIG)


GPT2_LARGE_CONFIG = {
    "vocab_size": 50257,     # Vocabular size
    "context_length": 1024,  # Context length
    "max_new_toknes": 10,    # Maximum new tokens to generate
    "embed_dim": 1280,       # Embedding dimension
    "d_ff": 4 * 1280,        # Hidden dimension
    "n_heads": 20,           # Number of attention heads
    "n_layers": 36,          # Number of transformer layers
    "dropout": 0.1,          # Dropout rate
    "qkv_bias": False,       # Query-Key-Value bias
    "dtype": torch.float32,
    "kv_window_size": 1024,  # KV cache window size
}
GPT2_LARGE_CONFIG= DotDict(GPT2_LARGE_CONFIG)


# 1.5B(1558M) GPT2
GPT2_XL_CONFIG = {
    "vocab_size": 50257,     # Vocabular size
    "context_length": 1024,  # Context length
    "max_new_toknes": 10,    # Maximum new tokens to generate
    "embed_dim": 1600,       # Embedding dimension
    "d_ff": 4 * 1600,        # Hidden dimension
    "n_heads": 25,           # Number of attention heads
    "n_layers": 48,          # Number of transformer layers
    "dropout": 0.1,          # Dropout rate
    "qkv_bias": False,       # Query-Key-Value bias
    "dtype": torch.float32,
    "kv_window_size": 1024,  # KV cache window size
}
GPT2_XL_CONFIG= DotDict(GPT2_XL_CONFIG)


# ##############################
# LLaMA
# ##############################
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


# ##############################
# tokenizer
# ##############################
tokenizer = tiktoken.get_encoding("gpt2")




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
