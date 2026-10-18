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

# python libraries
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)

import torch

from utils.args_tools import DotDict

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


# 注意：本模块只放静态配置，import 时不允许有副作用
# （device 探测 / 随机种子 / tokenizer 下载请在调用方获取）。

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
    "max_new_tokens": 200,   # Maximum new tokens to generate
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
    "max_new_tokens": 10,    # Maximum new tokens to generate
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
    "max_new_tokens": 10,    # Maximum new tokens to generate
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
    "max_new_tokens": 10,    # Maximum new tokens to generate
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




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
