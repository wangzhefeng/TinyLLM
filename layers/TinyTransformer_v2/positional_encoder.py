# -*- coding: utf-8 -*-

# ***************************************************
# * File        : positional_encoder.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-01-01
# * Version     : 1.0.010114
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

import torch
import torch.nn as nn

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


class PositionalEncoder(nn.Module):

    def __init__(self, d_model = 512, max_seq_len = 80):
        """
        Args:
            d_model (_type_): 词嵌入维度
            max_seq_len (int, optional): 最大序列长度. Defaults to 80.
        """
        super().__init__()
        
        self.d_model = d_model
        # 根据 pos 和 i 创建一个常量 PE 矩阵
        pe = torch.zeros(max_seq_len, d_model)  # size: (max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** (i / d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** (i / d_model)))
        pe = pe.unsqueeze(0)  # size: (1, max_seq_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # 使得单词嵌入表示相对大一些
        x = x * math.sqrt(self.d_model)
        # 增加位置常量到单词嵌入表示中
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :].require_grad_(False)
        return x




# 测试代码 main 函数
def main():
    # params
    d_model = 2
    max_seq_len = 3
    
    # positional encoding
    pe = torch.zeros(max_seq_len, d_model)
    for pos in range(max_seq_len):
        for i in range(0, d_model, 2):
            pe[pos, i] = math.sin(pos / (10000 ** (i / d_model)))
            pe[pos, i + 1] = math.cos(pos / (10000 ** (i / d_model)))
    print(pe)
    print(pe.size())
    
    pe = pe.unsqueeze(0)
    print(pe)
    print(pe.size())
    
    x = torch.ones(1, max_seq_len, d_model)
    print(f"x size: {x.size(1)}")
    
    x = x + pe[:, :x.size(1)]
    print(f"x: \n{x}")
    
    print(pe[:, :x.size(1)])
    print(pe[:, :x.size(1), :])

if __name__ == "__main__":
    main()
