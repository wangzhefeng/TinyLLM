# -*- coding: utf-8 -*-

# ***************************************************
# * File        : model_config.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-08-30
# * Version     : 1.0.083007
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

__all__ = []

# python libraries
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)
import warnings
warnings.filterwarnings("ignore")

import tiktoken
import torch

from utils.args_tools import DotDict
from utils.device import device_setting

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


# device
device = device_setting()

# model random seed
torch.manual_seed(123)

# model params
GPT2_124M_CONFIG = {
    "vocab_size": 50257,     # Vocabular size
    "context_length": 1024,  # Context length
    "max_new_toknes": 200,    # Maximum new tokens to generate
    "embed_dim": 768,        # Embedding dimension
    "d_ff": 4 * 768,         # Hidden dimension
    "n_heads": 12,           # Number of attention heads
    "n_layers": 12,          # Number of transformer layers
    "dropout": 0.1,          # Dropout rate
    "qkv_bias": False,       # Query-Key-Value bias
    "dtype": torch.float32,
    "kv_window_size": 1024,  # KV cache window size
}
GPT2_124M_CONFIG= DotDict(GPT2_124M_CONFIG)


GPT2_MEDIUM_CONFIG = {
    "vocab_size": 50257,     # Vocabular size
    "context_length": 1024,  # Context length
    "max_new_toknes": 10,    # Maximum new tokens to generate
    "embed_dim": 1024,        # Embedding dimension
    "d_ff": 4 * 1024,         # Hidden dimension
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
    "embed_dim": 1280,        # Embedding dimension
    "d_ff": 4 * 1280,         # Hidden dimension
    "n_heads": 20,           # Number of attention heads
    "n_layers": 36,          # Number of transformer layers
    "dropout": 0.1,          # Dropout rate
    "qkv_bias": False,       # Query-Key-Value bias
    "dtype": torch.float32,
    "kv_window_size": 1024,  # KV cache window size
}
GPT2_LARGE_CONFIG= DotDict(GPT2_LARGE_CONFIG)


GPT2_XL_CONFIG = {
    "vocab_size": 50257,     # Vocabular size
    "context_length": 1024,  # Context length
    "max_new_toknes": 10,    # Maximum new tokens to generate
    "embed_dim": 1600,        # Embedding dimension
    "d_ff": 4 * 1600,         # Hidden dimension
    "n_heads": 25,           # Number of attention heads
    "n_layers": 48,          # Number of transformer layers
    "dropout": 0.1,          # Dropout rate
    "qkv_bias": False,       # Query-Key-Value bias
    "dtype": torch.float32,
    "kv_window_size": 1024,  # KV cache window size
}
GPT2_XL_CONFIG= DotDict(GPT2_XL_CONFIG)


# tokenizer
tokenizer = tiktoken.get_encoding("gpt2")




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
