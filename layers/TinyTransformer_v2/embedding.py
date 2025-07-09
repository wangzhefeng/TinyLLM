# -*- coding: utf-8 -*-

# ***************************************************
# * File        : embedding.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-01-01
# * Version     : 1.0.010115
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

__all__ = []

# python libraries
import os
import sys
ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import torch
import torch.nn as nn

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


class Embedder(nn.Module):
    """
    词嵌入层，将token索引转换为向量表示
    Args:
        vocab_size (int): 词汇表大小
        d_model (int): 嵌入维度
    """
    
    def __init__(self, vocab_size, d_model):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model
    
    def forward(self, x):
        """
        前向传播
        Args:
            x (torch.Tensor): 输入token索引，shape: (batch_size, seq_len)
        Returns:
            torch.Tensor: 词嵌入向量，shape: (batch_size, seq_len, d_model)
        """
        return self.embedding(x) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
