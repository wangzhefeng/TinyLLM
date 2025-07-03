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
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)

import torch

from utils.log_util import logger

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


def device_setting(verbose: bool = False):
    """
    device setting
    """
    if verbose:
        logger.info(f"{40 * '='}")
        logger.info(f"Device Info:")
        logger.info(f"{40 * '='}")
        logger.info(f"GPU available: {torch.cuda.is_available() or torch.backends.mps.is_available()}")
        logger.info(f"GPU count: {torch.cuda.device_count()}")
    
    if torch.cuda.is_available():
        if verbose:
            logger.info(f"current GPU name: {torch.cuda.get_device_name()}")
            logger.info(f"current GPU id: {torch.cuda.current_device()}")
        gpu = torch.cuda.current_device()
        # torch.cuda.set_device(0)
        device = torch.device(f"cuda{gpu}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    if verbose:
        logger.info(f"Using device: {device.type.upper()}")

    return device


# TODO
def _acquire_device(use_gpu: bool=True, gpu_type: str="cuda", use_multi_gpu: bool=False, devices: str="0,1,2,3,4,5,6,7"):
    # use gpu or not
    use_gpu = True \
        if use_gpu and (torch.cuda.is_available() or torch.backends.mps.is_available()) \
        else False
    # gpu type: "cuda", "mps"
    gpu_type = gpu_type.lower().strip()
    # gpu device ids list
    devices = devices.replace(" ", "")
    device_ids = [int(id_) for id_ in devices.split(",")]
    # gpu device ids string
    gpu = device_ids[0]
    # device
    if use_gpu and gpu_type == "cuda":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu) if not use_multi_gpu else devices
        device = torch.device(f"cuda:{gpu}")
        logger.info(f"Use device GPU: cuda:{gpu}")
    elif use_gpu and gpu_type == "mps":
        device = torch.device("mps") \
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() \
            else torch.device("cpu")
        logger.info(f"Use device GPU: mps")
    else:
        device = torch.device("cpu")
        logger.info("Use device CPU")

    return device


def torch_gc(device_id=""):
    """
    清理 GPU 内存函数
    
    device_id: # CUDA 设备 ID，如果未设置则为空
    """
    # 设置设备参数
    if torch.cuda.is_available():
        DEVICE_TYPE = torch.device("cuda")
    elif torch.backends.mps.is_available():
        DEVICE_TYPE = torch.device("mps")
    else:
        DEVICE_TYPE = torch.device("cpu")  # 使用 CUDA
    DEVICE = f"{DEVICE_TYPE}:{device_id}" \
        if DEVICE_TYPE == torch.device("cuda") and device_id \
        else DEVICE_TYPE  # 组合 CUDA 设备信息

    if torch.cuda.is_available():  # 检查是否可用 CUDA
        with torch.cuda.device(DEVICE):  # 指定 CUDA 设备
            torch.cuda.empty_cache()  # 清空 CUDA 缓存
            torch.cuda.ipc_collect()  # 收集 CUDA 内存碎片
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()




# 测试代码 main 函数
def main():
    device = device_setting(verbose=True)

if __name__ == "__main__":
    main()
