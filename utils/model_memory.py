# -*- coding: utf-8 -*-

# ***************************************************
# * File        : model_memory.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-06-09
# * Version     : 1.0.060914
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

__all__ = []

# python libraries
import os
import sys
ROOT = str(os.getcwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)
from warnings import simplefilter
simplefilter("ignore")
from pathlib import Path

import torch

from utils.log_util import logger

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


def model_memory_size(model, input_dtype=torch.float32, verbose:bool=False):
    """
    calculate the memory requirements for this model
    """
    # calculate total number of elements per parameter
    total_params = sum([
        param.numel() 
        # param.nelement() 
        for param in model.parameters()
    ])
    # check if gradients are stored for this parameter
    total_grads = sum([
        param.numel() 
        for param in model.parameters() 
        if param.requires_grad
    ])
    # calculate buffer size(non-parameters that require memory)
    total_buffers = sum([
        buffer.numel() 
        for buffer in model.buffers()
    ])
    if verbose:
        logger.info(f"Model number of parameters: {total_params / 1e6:.2f}M.")
        # logger.info(f"Total number of parameters: {total_params + total_grads + total_buffers}.")

    # Size in bytes = (Number of elements) * (Size of each element in bytes)
    # assume parameters and gradients are stored in the same type as input dtype
    element_size = torch.tensor(0, dtype=input_dtype).element_size()
    total_memory_bytes = (total_params + total_grads + total_buffers) * element_size
    # convert bytes to gigabytes
    total_memory_gb = total_memory_bytes / (1024 ** 3)
    if verbose:
        logger.info(f"Model memory used size: {total_memory_gb:.2f}GB.")

    return total_memory_gb




# 测试代码 main 函数
def main():
    from utils.args_tools import DotDict
    from models.llama2 import Model

    # model params
    LLAMA2_CONFIG_7B = {
        "vocab_size": 32000,     # Vocabulary size
        "context_length": 4096,  # Context length
        "emb_dim": 4096,         # Embedding dimension
        "n_heads": 32,           # Number of attention heads
        "n_layers": 32,          # Number of layers
        "hidden_dim": 11008,     # NEW: Size of the intermediate dimension in FeedForward
        "dtype": torch.bfloat16  # NEW: Lower-precision dtype to reduce memory usage
    }
    LLAMA2_CONFIG_7B = DotDict(LLAMA2_CONFIG_7B)

    # model
    model = Model(LLAMA2_CONFIG_7B)

    # model memory size
    total_memory_gb = model_memory_size(model, input_dtype=torch.float32)
    total_memory_gb = model_memory_size(model, input_dtype=torch.bfloat16)

if __name__ == "__main__":
    main()
