# -*- coding: utf-8 -*-

# ***************************************************
# * File        : attention.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-03-29
# * Version     : 1.0.032923
# * Description : description
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

import torch
import torch.nn as nn

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


class CasualAttention(nn.Module):
    """
    Casual Self Attention
    """
    def __init__(self, d_in: int, d_out: int, context_length: int, 
                 dropout: float, qkv_bias=False):
        super().__init__()

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
        self.register_buffer("mask", mask)
    
    def forward(self, x):
        batch_size, num_tokens, d_in = x.shape
        # q, k, v
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)
        # attention scores
        attn_scores = queries @ keys.transpose(1, 2)
        # casual mask
        attn_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        # attention weights
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        # dropout
        attn_weights = self.dropout(attn_weights)
        # context scores
        context_vec = attn_weights @ values
    
        return context_vec


class MultiHeadAttentionWrapper(nn.Module):
    
    def __init__(self, d_in: int, d_out: int, context_length: int, 
                 dropout: float, num_heads: int, qkv_bias=False):
        super().__init__()
        
        self.heads = nn.ModuleList([
            CasualAttention(d_in, d_out, context_length, dropout, qkv_bias) 
            for _ in range(num_heads)
        ])
        self.out_proj = nn.Linear(d_out * num_heads, d_out * num_heads)

    def forward(self, x):
        context_vector = torch.cat([head(x) for head in self.heads], dim=-1)
        projection = self.out_proj(context_vector)

        return projection




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
