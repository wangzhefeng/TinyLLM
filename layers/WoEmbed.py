# -*- coding: utf-8 -*-

# ***************************************************
# * File        : WoE.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-07-09
# * Version     : 1.0.070922
# * Description : Word Embedding
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

__all__ = []

# python libraries
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)
import math
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


class Embeddings(nn.Module):

    def __init__(self, d_model, vocab_size):
        super().__init__()

        self.lut = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model
    
    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
