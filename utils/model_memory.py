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
    "model_param_counts",
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


def _is_weight_tied(model, token_embedding):
    """检测是否存在真实的 weight tying：除 token embedding 自身外，
    还有其他模块的参数与 ``token_embedding.weight`` 是同一存储。"""
    emb_weight = token_embedding.weight
    emb_ptr = emb_weight.data_ptr()
    for module in model.modules():
        if module is token_embedding:
            continue
        for param in module.parameters(recurse=False):
            if param is emb_weight or param.data_ptr() == emb_ptr:
                return True

    return False


def model_param_counts(model):
    """统计模型参数量（供 model_memory_size 与测试复用）。

    Returns:
        dict:
            total_params: 注册的参数总量（weight tying 时共享张量按槽位重复计数）
            total_grads: 需要梯度的参数量（共享张量只存一份梯度，去重计数）
            total_buffers: buffer 总量
            total_params_normalized: 去重后的唯一参数量；
                仅在检测到真实 weight tying 时扣除一份 token embedding
            weight_tied: 是否检测到输出头与 token embedding 共享权重
    """
    # 注册槽位计数：remove_duplicate=False 使 tying 的共享张量被重复计入
    total_params = sum(
        param.numel()
        for _, param in model.named_parameters(remove_duplicate=False)
    )
    # 梯度按存储计数：parameters() 对共享张量只产出一次
    total_grads = sum(
        param.numel() for param in model.parameters() if param.requires_grad
    )
    total_buffers = sum(buffer.numel() for buffer in model.buffers())

    # weight tying：仅当输出头确实与 token embedding 共享权重时才扣除 embedding；
    # 未 tying 的模型（如 GPT2：tok_embed 与 out_head 各自独立）不做扣除
    token_embedding = _get_token_embedding(model)
    weight_tied = token_embedding is not None and _is_weight_tied(model, token_embedding)
    total_params_normalized = total_params - (
        token_embedding.weight.numel() if weight_tied else 0
    )

    return {
        "total_params": total_params,
        "total_grads": total_grads,
        "total_buffers": total_buffers,
        "total_params_normalized": total_params_normalized,
        "weight_tied": weight_tied,
    }


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
    counts = model_param_counts(model)
    total_params = counts["total_params"]
    total_grads = counts["total_grads"]
    total_buffers = counts["total_buffers"]
    total_params_normalized = counts["total_params_normalized"]

    # Size in bytes = (Number of elements) * (Size of each element in bytes)
    # 显存按物理存储估算：weight tying 的共享张量只占一份，使用去重后的参数量
    element_size = torch.tensor(0, dtype=input_dtype).element_size()
    total_memory_gb = (total_params_normalized + total_grads + total_buffers) * element_size / (1024 ** 3)

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
        logger.info(f"Total number of parameters: {(total_params_normalized + total_grads + total_buffers):,}")
        logger.info(f"Total number of parameters: {(total_params_normalized + total_grads + total_buffers) / 1e6:.2f}M.")
        logger.info(f"Weight tying (out head shares token embedding): {counts['weight_tied']}")
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
