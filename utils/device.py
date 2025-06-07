# -*- coding: utf-8 -*-

# ***************************************************
# * File        : device.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2025-02-09
# * Version     : 0.1.020915
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

__all__ = [
    "device_setting",
    "torch_gc",
]

# python libraries
import os
import sys
ROOT = str(os.getcwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)

import torch

from utils.log_util import logger

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


def device_setting(verbose: bool = False):
    """
    device setting
    """
    if verbose:
        logger.info(f"{50 * '='}")
        logger.info(f"Device Info:")
        logger.info(f"{50 * '='}")
        logger.info(f"GPU available: {torch.cuda.is_available()}")
        logger.info(f"GPU count: {torch.cuda.device_count()}")
    
    if torch.cuda.is_available():
        if verbose:
            logger.info(f"current GPU name: {torch.cuda.get_device_name()}")
            logger.info(f"current GPU id: {torch.cuda.current_device()}")
        # torch.cuda.set_device(0)
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    if verbose:
        logger.info(f"Using device: {device.type.upper()}.")

    return device


def torch_gc(gpu_type: str = "cuda", device: str = "cuda:0"):
    """
    empty cuda cache and memory pecices
    """
    if device != torch.device("cpu"):
        if gpu_type == "cuda":
            with torch.cuda.device(device):  # 指定 CUDA 设备
                torch.cuda.empty_cache()  # 清空 CUDA 缓存
                torch.cuda.ipc_collect()  # 收集 CUDA 内存碎片
        elif gpu_type == "mps":
            torch.backends.mps.empty_cache()


def torch_gc_v1():
    """
    清理 GPU 内存函数
    """
    # 设置设备参数
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 使用 CUDA
    DEVICE_ID = "0"  # CUDA 设备 ID，如果未设置则为空
    CUDA_DEVICE = f"{DEVICE}:{DEVICE_ID}" if DEVICE_ID else DEVICE  # 组合 CUDA 设备信息
    logger.info(f"cuda device: {CUDA_DEVICE}")

    if torch.cuda.is_available():  # 检查是否可用 CUDA
        with torch.cuda.device(CUDA_DEVICE):  # 指定 CUDA 设备
            torch.cuda.empty_cache()  # 清空 CUDA 缓存
            torch.cuda.ipc_collect()  # 收集 CUDA 内存碎片




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
