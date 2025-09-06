# -*- coding: utf-8 -*-

# ***************************************************
# * File        : hparam_search.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-09-07
# * Version     : 1.0.090700
# * Description : description
# * Link        : https://github.com/rasbt/LLMs-from-scratch/blob/main/ch05/05_bonus_hparam_tuning/hparam_search.py
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

__all__ = []

# python libraries
import os
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)
import math
import itertools
import warnings
warnings.filterwarnings("ignore")

import torch

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]
os.environ['LOG_NAME'] = LOGGING_LABEL
from utils.log_util import logger


# Define a grid of hyperparameters to search over
HPARAM_GRID = {
    "batch_size": [2, 4, 8, 16],
    "drop_rate": [0.0, 0.1, 0.2],
    "warmup_iters": [10, 20, 30],
    "weight_decay": [0.0, 0.1, 0.01],
    "peak_lr": [0.001, 0.005, 0.0001, 0.0005],
    "initial_lr": [0.0001, 0.00005],
    "min_lr": [0.0001, 0.00001, 0.00001],
    "n_epochs": [5, 10, 15, 20, 25],
}







# 测试代码 main 函数
def main():
    # Generate all combinations of hyperparameters
    pass

if __name__ == "__main__":
    main()
