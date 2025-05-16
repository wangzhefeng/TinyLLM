# -*- coding: utf-8 -*-

# ***************************************************
# * File        : DyT.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-05-01
# * Version     : 1.0.050119
# * Description : description
# * Link        : https://arxiv.org/pdf/1910.07467
# *               https://jiachenzhu.github.io/DyT/
# *               https://mp.weixin.qq.com/s/oEVuQ9hQEKa53EFfSMBrLg
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# * TODO        : 1.
# ***************************************************

__all__ = []

# python libraries
import os
import sys
ROOT = str(os.getcwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)

import torch
import torch.nn as nn

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


class DyT(nn.Module):
    
    def __init__(self, num_features, alpha_init_value=0.5):
        super().__init__()

        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
    
    def forward(self, x):
        x = torch.tanh(self.alpha * x)
        
        return x * self.weight + self.bias




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
