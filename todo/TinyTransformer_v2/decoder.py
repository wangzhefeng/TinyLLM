# -*- coding: utf-8 -*-

# ***************************************************
# * File        : decoder.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-01-01
# * Version     : 1.0.010116
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

import torch
import torch.nn as nn

from layer_norm import Norm
from multi_head_attention import MultiHeadAttention
from feed_forward import FeedForward
from tiny_model.tiny_transformer.embedding import Embedder
from positional_encoder import PositionalEncoder

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


class DecoderLayer(nn.Module):
    
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        
        self.norm1 = Norm(d_model)
        self.norm2 = Norm(d_model)
        self.norm3 = Norm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.attn1 = MultiHeadAttention(d_model, heads, dropout=dropout)
        self.attn2 = MultiHeadAttention(d_model, heads, dropout=dropout)
        self.ff = FeedForward(d_model, d_ff=2048, dropout=dropout)
    
    def forward(self, x, e_outputs, src_mask, trg_mask):
        # masked multi-head attention
        attn_output_1 = self.attn1(x, x, x, trg_mask)
        attn_output_1 = self.dropout1(attn_output_1)
        # add & norm
        x = x + attn_output_1
        x = self.norm1(x)

        # multi-head attention
        attn_output_2 = self.attn2(x, e_outputs, e_outputs, src_mask)
        attn_output_2 = self.dropout2(attn_output_2)
        # add & norm
        x = x + attn_output_2
        x = self.norm2(x)

        # feed forward
        ff_output = self.ff(x)
        ff_output = self.dropout3(ff_output)
        # add & norm
        x = x + ff_output
        x = self.norm3(x)
        
        return x


class Decoder(nn.Module):
    
    def __init__(self, vocab_size, d_model,  N, heads, dropout):
        super().__init__()

        self.N = N
        self.embed = Embedder(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model, max_seq_len=80, dropout=dropout)
        self.layers = [DecoderLayer(d_model, heads, dropout) for _ in range(N)]
        self.norm = Norm(d_model)

    def forward(self, trg, e_outputs, src_mask, trg_mask):
        x = self.embed(trg)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, e_outputs, src_mask, trg_mask)
        x = self.norm(x)
        
        return x


# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
