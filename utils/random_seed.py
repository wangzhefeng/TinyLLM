# -*- coding: utf-8 -*-

# ***************************************************
# * File        : seed.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2025-02-28
# * Version     : 0.1.022823
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

__all__ = [
    "set_seed_ml",
    "set_seed",
]

# python libraries
import os
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
    设置可重复随机数
    """
    random.seed(seed)
    np.random.seed(seed)


def set_seed(seed: int = 2025):
    """
    设置可重复随机数
    manual_seed: https://docs.pytorch.org/docs/stable/generated/torch.cuda.manual_seed.html
    manual_seed_all: https://docs.pytorch.org/docs/stable/generated/torch.cuda.manual_seed_all.html
    """
    random.seed(seed)
    np.random.seed(seed)
    # Sets the seed for generating random numbers on all devices. Returns a torch.Generator object.
    torch.manual_seed(seed)
    # Set the seed for generating random numbers for the current GPU.
    torch.cuda.manual_seed(seed)
    # Set the seed for generating random numbers on all GPUs.
    torch.cuda.manual_seed_all(seed)


def set_cudnn():
    torch.backends.cudnn.deterministic = True




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
