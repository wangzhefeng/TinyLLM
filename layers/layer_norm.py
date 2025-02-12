# -*- coding: utf-8 -*-

# ***************************************************
# * File        : layer_norm.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-02-12
# * Version     : 1.0.021221
# * Description : description
# * Link        : link
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


class LayerNorm(nn.Module):
    
    def __init__(self, emb_dim, eps = 1e-5):
        super().__init__()

        self.eps = eps
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift






# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
