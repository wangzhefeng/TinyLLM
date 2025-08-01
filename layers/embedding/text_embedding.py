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

from utils.log_util import logger

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


class TokenEmbeddings(nn.Module):

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()

        self.lut = nn.Embedding(vocab_size, d_model)
        # logger.info(f"debug::self.lut.weight: \n{self.lut.weight}")
        self.d_model = d_model
    
    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class AbsolutePositionEmbeddings(nn.Module):
    
    def __init__(self, d_model: int, context_length: int):
        super().__init__()

        self.context_length = context_length
        self.pe = nn.Embedding(context_length, d_model)
    
    def forward(self, x):
        assert x == torch.arange(self.context_length)
        return self.pe(x)




# 测试代码 main 函数
def main():
    max_context_length = 100
    token_embed = TokenEmbeddings(d_model=512, vocab_size=1000)
    pos_embed = AbsolutePositionEmbeddings(d_model=512, context_length=100)
    
    dataloader = None
    for batch in dataloader:
        x, y = batch
        input_embed = token_embed(x) + pos_embed(torch.arange(max_context_length))
        print(input_embed)

if __name__ == "__main__":
    main()
