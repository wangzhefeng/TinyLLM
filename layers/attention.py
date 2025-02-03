# -*- coding: utf-8 -*-

# ***************************************************
# * File        : attention.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-01-25
# * Version     : 1.0.012513
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

from utils.log_util import logger

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


class CasualAttention(nn.Module):
    """
    Casual Self Attention
    """
    def __init__(self, d_in: int, d_out: int, context_length: int, dropout, qkv_bias=False):
        super().__init__()

        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = torch.nn.Dropout(dropout)
        mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
        self.register_buffer("mask", mask)
    
    def forward(self, x):
        b, num_tokens, d_in = x.shape
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
        context_vec = attn_weights @ values
    
        return context_vec


class MultiHeadAttentionWrapper(nn.Module):
    
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        
        self.heads = nn.ModuleList([
            CasualAttention(d_in, d_out, context_length, dropout, qkv_bias) 
            for _ in range(num_heads)
        ])

    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)


class MultiHeadAttention(nn.Module):
    
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()

        assert (d_out % num_heads == 0), "d_out must be divisible by num_heads"
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        # query, key, value weights
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        # Linear layer to combine head outputs
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))
    
    def forward(self, x):
        b, num_tokens, d_in = x.shape
        # shape: [b, num_tokens, d_out]
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        # split the matrix by adding a "num_heads" dimension, 
        # unroll last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        # transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)
        # compute scaled dot-product attention(aka self-attention) with a causal mask
        attn_scores = queries @ keys.transpose(2, 3)  # dot product for each head
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        # mask to fill attention scores
        attn_scores.masked_fill_(mask_bool, -torch.inf)
        # compute attention weights
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)
        # context vector, shape: (b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2)
        # combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        # optional projection
        context_vec = self.out_proj(context_vec)

        return context_vec






# 测试代码 main 函数
def main():
    # input text
    input_strings = "Your journey starts with one step."

    # tokens embeddings
    inputs = torch.tensor(
        [[0.43, 0.15, 0.89], # Your     (x^1)
        [0.55, 0.87, 0.66], # journey  (x^2)
        [0.57, 0.85, 0.64], # starts   (x^3)
        [0.22, 0.58, 0.33], # with     (x^4)
        [0.77, 0.25, 0.10], # one      (x^5)
        [0.05, 0.80, 0.55]] # step     (x^6)
    )
    logger.info(f"inputs: \n{inputs}")
    logger.info(f"inputs.shape: {inputs.shape}")

    # the input embedding size, d=3
    d_in = inputs.shape[1]
    # the output embedding size, d=2
    d_out = 2

    # batch token embedding
    batch = torch.stack((inputs, inputs), dim=0)
    logger.info(f"batch: \n{batch}")
    logger.info(f"batch.shape: {batch.shape}")
    
    # ------------------------------
    # method 1
    # ------------------------------
    # attention
    torch.manual_seed(123)
    context_length = batch.shape[1]
    casual_attn = CasualAttention(d_in, d_out, context_length, dropout=0)
    context_vecs = casual_attn(batch)
    logger.info(f"context_vecs: \n{context_vecs}")
    logger.info(f"context_vecs.shape: {context_vecs.shape}")
    
    # multi-head attention
    torch.manual_seed(123)
    context_length = batch.shape[1]
    mha = MultiHeadAttentionWrapper(d_in, d_out, context_length, dropout=0.0, num_heads=2)
    context_vecs = mha(batch)
    logger.info(f"context_vecs: \n{context_vecs}")
    logger.info(f"context_vecs.shape: \n{context_vecs.shape}")
    # ------------------------------
    # method 2
    # ------------------------------
    torch.manual_seed(123)
    batch_size, context_length, d_in = batch.shape
    d_out = 2
    mha = MultiHeadAttention(d_in, d_out, context_length, 0.0, num_heads=2)
    context_vecs = mha(batch)
    print(context_vecs)
    print(f"context_vecs.shape: {context_vecs.shape}")

if __name__ == "__main__":
    main()
