# -*- coding: utf-8 -*-

# ***************************************************
# * File        : models.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-01-02
# * Version     : 1.0.010223
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# * TODO        : 1.
# ***************************************************

__all__ = []

# python libraries
import os
import sys
ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import torch.nn as nn

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


class DeepLab(nn.Module):

    def __init__(self):
        super(DeepLab, self).__init__()
    
    def forward(self, x):
        return x





# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
