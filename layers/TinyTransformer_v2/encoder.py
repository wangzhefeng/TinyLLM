# -*- coding: utf-8 -*-

# ***************************************************
# * File        : encoder_layer.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-01-01
# * Version     : 1.0.010116
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

import torch
import torch.nn as nn

from layer_norm import Norm
from multi_head_attention import MultiHeadAttention
from feed_forward import FeedForward
from tiny_model.tiny_transformer.embedding import Embedder
from positional_encoder import PositionalEncoder

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


class EncoderLayer(nn.Module):
    
    def __init__(self, d_model, heads, dropout = 0.1):
        super().__init__()

        self.norm1 = Norm(d_model)
        self.norm2 = Norm(d_model)
        self.attn = MultiHeadAttention(d_model, heads, dropout=dropout)
        self.ff = FeedForward(d_model, d_ff=2048, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x, mask):
        # multi-head attention
        attn_output = self.attn(x, x, x, mask)
        attn_output = self.dropout1(attn_output)
        # add & norm
        x = x + attn_output
        x = self.norm1(x)

        # feed forward
        ff_output = self.ff(x)
        ff_output = self.dropout2(ff_output)
        # add & norm
        x = x + ff_output
        x = self.norm2(x)

        return x


class Encoder(nn.Module):
    
    def __init__(self, vocab_size, d_model, N, heads, dropout):
        super().__init__()

        self.N = N
        self.embed = Embedder(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model, max_seq_len=80, dropout=dropout)
        self.layers = [EncoderLayer(d_model, heads, dropout) for _ in range(N)]
        self.norm = Norm(d_model)
    
    def forward(self, src, mask):
        x = self.embed(src)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, mask)
        x = self.norm(x)
        
        return x




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
