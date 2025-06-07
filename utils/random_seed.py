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
    "set_seed",
]

# python libraries
import os
import sys
ROOT = str(os.getcwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)
import random

import numpy as np
import torch

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


def set_seed_ml(seed: int = 2025):
    """
    设置可重复随机数
    """
    random.seed(seed)
    np.random.seed(seed)


def set_seed(seed: int = 2025):
    """
    设置可重复随机数
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
