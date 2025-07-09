# -*- coding: utf-8 -*-

# ***************************************************
# * File        : multi_head_attention.py
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
import torch.nn.functional as F

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


class MultiHeadAttention(nn.Module):
    
    def __init__(self, d_model, heads, dropout = 0.1):
        super(MultiHeadAttention, self).__init__()
        # params
        self.d_model = d_model
        self.h = heads
        self.d_k = d_model // heads
        # layers
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    def attention(self, q, k, v, d_k, mask=None, dropout=None):
        # QK/sqrt(d_k)
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
        # 掩盖掉那些为了填补长度增加的单元，使其通过 softmax 计算后为 0
        if mask is not None:
            mask = mask.unsequeeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        # softmax(QK/sqrt(d_k))
        scores = F.softmax(scores, dim=-1)
        # dropout
        if dropout is not None:
            scores = dropout(scores)
        # softmax(QK/sqrt(d_k))V
        output = torch.matmul(scores, v)

        return output

    def forward(self, q, k, v, mask=None):
        # batch size
        bs = q.size(0)
        # 进行线性操作划分成 h 个头, 并进行矩阵转置
        # q: (bs, nq, d_model) -> (bs, nq, h, d_k) -> (bs, h, nq, d_k)
        # k: (bs, nk, d_model) -> (bs, nk, h, d_k) -> (bs, h, nk, d_k)
        # v: (bs, nk, d_model) -> (bs, nk, h, d_k) -> (bs, h, nk, d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k).transpose(1, 2)
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k).transpose(1, 2)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k).transpose(1, 2)
        # 计算 attention
        scores = self.attention(q, k, v, self.d_k, mask, self.dropout)
        # 链接多个头，并输出到最后的线性层
        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        output = self.out(concat)

        return output




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
