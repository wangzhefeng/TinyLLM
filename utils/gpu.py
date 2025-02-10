# -*- coding: utf-8 -*-

# ***************************************************
# * File        : gpu.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2024-11-11
# * Version     : 0.1.111122
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import os
import sys
ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import torch

from utils.log_util import logger

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


# 设置设备参数
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 使用 CUDA
DEVICE_ID = "0"  # CUDA 设备 ID，如果未设置则为空
CUDA_DEVICE = f"{DEVICE}:{DEVICE_ID}" if DEVICE_ID else DEVICE  # 组合 CUDA 设备信息
logger.info(f"cuda device: {CUDA_DEVICE}")


def torch_gc():
    """
    清理 GPU 内存函数
    """
    if torch.cuda.is_available():  # 检查是否可用 CUDA
        with torch.cuda.device(CUDA_DEVICE):  # 指定 CUDA 设备
            torch.cuda.empty_cache()  # 清空 CUDA 缓存
            torch.cuda.ipc_collect()  # 收集 CUDA 内存碎片




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
