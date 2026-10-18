# -*- coding: utf-8 -*-

# ***************************************************
# * File        : random_seed.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-02-04
# * Version     : 1.0.020400
# * Description : reproducible random seed helpers
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

__all__ = [
    "set_seed_ml",
    "set_seed",
    "set_seed_torch",
    "set_seed_worker",
    "set_cudnn",
]

# python libraries
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)

import random

import numpy as np
import torch

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


def set_seed_ml(seed: int = 2025):
    """
    设置 python / numpy 随机数种子
    """
    random.seed(seed)
    np.random.seed(seed)


def set_seed(seed: int = 2025):
    """
    设置可重复随机数（python / numpy / torch，含 CUDA）
    """
    random.seed(seed)
    np.random.seed(seed)
    # CPU / all devices
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        # current GPU
        torch.cuda.manual_seed(seed)
        # all GPUs
        torch.cuda.manual_seed_all(seed)


def set_seed_torch(seed: int = 2025):
    """
    仅设置 torch（含 CUDA）随机数种子，保留给历史脚本使用
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def set_seed_worker(worker_id: int = 0):
    """
    DataLoader worker 的随机种子
    """
    seed = torch.initial_seed() % 2 ** 32
    set_seed_ml(seed)


def set_cudnn(deterministic: bool = True, benchmark: bool = False):
    """
    cuDNN 设置
    """
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = benchmark




# 测试代码 main 函数
def main():
    set_seed(2025)
    print(torch.rand(3))

if __name__ == "__main__":
    main()
