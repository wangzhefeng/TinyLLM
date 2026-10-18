# -*- coding: utf-8 -*-

# ***************************************************
# * File        : model_memory.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-06-09
# * Version     : 1.0.060914
# * Description : estimate model parameter / activation memory
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

__all__ = [
    "model_memory_size",
]

# python libraries
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)
import warnings
warnings.filterwarnings("ignore")

import torch

from utils.log_util import logger

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


def _get_token_embedding(model):
    """
    TinyLLM 中不同模型的 token embedding 命名不一致：
    - GPT2 / Qwen3 / Gemma3: tok_embed
    - Llama: tok_emb
    """
    for name in ("tok_embed", "tok_emb"):
        module = getattr(model, name, None)
        if module is not None and hasattr(module, "weight"):
            return module

    return None


def model_memory_size(model, input_dtype=torch.float32, verbose: bool = True):
    """
    calculate the memory requirements for this model

    Args:
        model: nn.Module
        input_dtype: dtype used to estimate element size
        verbose: print model info or not

    Returns:
        float: estimated total memory in GB
    """
    # parameters / gradients / buffers
    total_params = 0
    total_grads = 0
    for param in model.parameters():
        numel = param.numel()
        total_params += numel
        if param.requires_grad:
            total_grads += numel
    total_buffers = sum(buffer.numel() for buffer in model.buffers())

    # weight tying: output head may share weights with token embedding
    token_embedding = _get_token_embedding(model)
    total_params_normalized = total_params - (
        token_embedding.weight.numel() if token_embedding is not None else 0
    )

    # Size in bytes = (Number of elements) * (Size of each element in bytes)
    element_size = torch.tensor(0, dtype=input_dtype).element_size()
    total_memory_gb = (total_params + total_grads + total_buffers) * element_size / (1024 ** 3)

    # CUDA memory statistics are only available on CUDA devices
    allocated_memory = max_allocated_memory = cached_memory = None
    if torch.cuda.is_available():
        allocated_memory = torch.cuda.memory_allocated() / (1024 ** 3)
        max_allocated_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)
        cached_memory = torch.cuda.memory_reserved() / (1024 ** 3)

    if verbose:
        logger.info(f"{40 * '='}")
        logger.info("Model Info:")
        logger.info(f"{40 * '='}")
        logger.info(f"Model number of parameters: {total_params:,}")
        logger.info(f"Model number of parameters: {total_params / 1e6:.2f}M.")
        logger.info(f"Total number of parameters: {(total_params + total_grads + total_buffers):,}")
        logger.info(f"Total number of parameters: {(total_params + total_grads + total_buffers) / 1e6:.2f}M.")
        logger.info(f"Total number of unique parameters: {total_params_normalized:,}")
        logger.info(f"Model memory used size: {total_memory_gb:.2f}GB.")
        if allocated_memory is not None:
            logger.info(f"Allocated tensor memory: {allocated_memory:.2f}GB.")
            logger.info(f"Max allocated tensor memory: {max_allocated_memory:.2f}GB.")
            logger.info(f"Cached total memory: {cached_memory:.2f}GB.")
        logger.info(f"{40 * '='}")

    return total_memory_gb




# 测试代码 main 函数
def main():
    dtype_float_64 = torch.float64
    dtype_float_32 = torch.float32
    dtype_float_16 = torch.float16
    dtype_int_64 = torch.int64
    dtype_int_32 = torch.int32
    dtype_int_8 = torch.int8
    dtype_uint_8 = torch.uint8
    for dtype in (
        dtype_float_64,
        dtype_float_32,
        dtype_float_16,
        dtype_int_64,
        dtype_int_32,
        dtype_int_8,
        dtype_uint_8,
    ):
        element_size = torch.tensor(0.0, dtype=dtype).element_size()
        logger.info(f"{dtype} element size: {element_size}")

if __name__ == "__main__":
    main()
