# -*- coding: utf-8 -*-

# ***************************************************
# * File        : miniGPT.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-07-04
# * Version     : 1.0.070413
# * Description : description
# * Link        : link
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
from dataclasses import dataclass
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from torch.nn import functional as F

from utils.log_util import logger

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]








# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
