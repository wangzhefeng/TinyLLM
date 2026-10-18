# -*- coding: utf-8 -*-

# ***************************************************
# * File        : norm.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-07-27
# * Version     : 1.0.072716
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
import warnings
warnings.filterwarnings("ignore")

import torch.nn as nn

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]
os.environ['LOG_NAME'] = LOGGING_LABEL
from utils.log_util import logger


class RMSNorm(nn.Module):
    """
    用于注意力前后的层归一化
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        pass




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
