# -*- coding: utf-8 -*-

# ***************************************************
# * File        : transformer_add.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-07-12
# * Version     : 1.0.071222
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
import warnings
warnings.filterwarnings("ignore")
import math
import copy
import random
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]
os.environ['LOG_NAME'] = LOGGING_LABEL
from utils.log_util import logger


class TwoSumDataset(Dataset):
    
    def __init__(self, size = 100000):
        super().__init__()
        self.size = size
    
    @staticmethod
    def get_vocab():
        """
        定义字典
        """
        words_x = '<PAD>,1,2,3,4,5,6,7,8,9,0,<SOS>,<EOS>,+'
        vocab_x = {word: i for i, word in enumerate(words_x.split(','))}
        vocab_xr = [k for k, v in vocab_x.items()] #反查词典
        # print(f"vocab_x: \n{vocab_x}")
        # print(f"vocab_xr: \n{vocab_xr}")

        words_y = '<PAD>,1,2,3,4,5,6,7,8,9,0,<SOS>,<EOS>'
        vocab_y = {word: i for i, word in enumerate(words_y.split(','))}
        vocab_yr = [k for k, v in vocab_y.items()] #反查词典
        # print(f"vocab_y: \n{vocab_y}")
        # print(f"vocab_yr: \n{vocab_yr}")
        
        return vocab_x, vocab_xr, vocab_y, vocab_yr
    
    @staticmethod
    def encode_data(vocab_x: Dict, vocab_y: Dict):
        """
        两数相加数据集
        """
        # 定义词集合
        words = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        # 每个词被选中的概率
        p = np.array([7, 5, 5, 7, 6, 5, 7, 6, 5, 7])
        p = p / p.sum()
        # 随机采样 n1 个词作为 s1
        n1 = random.randint(10, 20)
        s1 = np.random.choice(words, size=n1, replace=True, p=p)
        s1 = s1.tolist()
        # 随机采样 n2 个词作为 s2
        n2 = random.randint(10, 20)
        s2 = np.random.choice(words, size=n2, replace=True, p=p)
        s2 = s2.tolist()
        # x等于s1和s2字符上的相加
        x = s1 + ['+'] + s2
        # y等于 s1 和 s2 数值上的相加
        y = int(''.join(s1)) + int(''.join(s2))
        y = list(str(y))
        # 加上首尾符号
        x = ['<SOS>'] + x + ['<EOS>']
        y = ['<SOS>'] + y + ['<EOS>']
        # 补pad到固定长度
        x = x + ['<PAD>'] * 50
        y = y + ['<PAD>'] * 51
        x = x[:50]
        y = y[:51]
        # 编码成 token
        token_x = [vocab_x[i] for i in x]
        token_y = [vocab_y[i] for i in y]
        # 转 tensor
        tensor_x = torch.LongTensor(token_x)
        tensor_y = torch.LongTensor(token_y)
        
        return tensor_x, tensor_y

    @staticmethod
    def decode_data(tensor_x, tensor_y, vocab_xr, vocab_yr) -> Tuple[str, str]:
        words_x = "".join([vocab_xr[i] for i in tensor_x.tolist()])
        words_y = "".join([vocab_yr[i] for i in tensor_y.tolist()])
        
        return words_x, words_y

    def __len__(self):
        return self.size
    
    def __getitem__(self, i):
        # 字典
        vocab_x, vocab_xr, vocab_y, vocab_yr = self.get_vocab()
        # encode
        x, y = self.encode_data(vocab_x, vocab_y) 
        # print(f"x: \n{x} \nx len: {len(x)}")
        # print(f"y: \n{y} \ny len: {len(y)}")
        # decode
        words_x, words_y = self.decode_data(x, y, vocab_xr, vocab_yr)
        # print(f"words_x: \n{words_x}")
        # print(f"words_y: \n{words_y}")
        
        return x, y


def create_dataloader():
    train_dl = DataLoader(
        dataset = TwoSumDataset(100000),
        batch_size = 200,
        drop_last = True,
        shuffle = True,
    )
    valid_dl = DataLoader(
        dataset = TwoSumDataset(10000),
        batch_size = 200,
        drop_last=True,
        shuffle = False,
    )
    
    return train_dl, valid_dl


def clones(module, N):
    """
    Produce N identical layers

    Args:
        module (_type_): _description_
        N (_type_): _description_
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class ScaledDotProductAttention(nn.Module):
    """
    Compute Scaled Dot-Product Attention
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, query, key, value, mask=None, dropout=None):
        d_k = query.size(-1)
        # attention
        scores = query@key.transpose(-2, -1) / math.sqrt(d_k)
        # mask attention
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e20)
        p_attn = F.softmax(scores, dim=-1)
        # dropout
        if dropout is not None:
            p_attn = dropout(p_attn)
        
        return p_attn@value, p_attn


class MultiHeadAttention(nn.Module):
    """
    Compute Multi-Head Attention
    """
    
    def __init__(self, n_head, d_model, dropout=0.1):
        super().__init__()

        assert d_model % n_head == 0

        # assume d_value always equals d_key
        self.d_k = d_model // n_head
        self.n_head = n_head
        self.linears = clones(
            nn.Linear(d_model, d_model), 4
        )
        # 记录 attention 矩阵结果
        self.attn = None
        self.dropout = nn.Dropout(dropout)
        self.attention = ScaledDotProductAttention()

    def forward(self, query, key, value, mask=None):
        # same mask applied to all h heads
        if mask is not None:
            mask = mask.unsqueeze(1)
        # number of batches
        n_batch = query.size(0)
        # do all the linear projections in batch from: d_model => n_head x d_k
        query, key, value = [
            linear(x).view(n_batch, -1, self.n_head, self.d_k).transpose(1, 2)
            for linear, x in zip(self.linears, (query, key, value))
        ]
        # apply attention on all the projected vectors in batch
        x, self.attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)
        # concat using a view and apply a final linear
        x = x.transpose(1, 2).contiguous().view(n_batch, -1, self.n_head * self.d_k)
        output = self.linears[-1](x)

        return output





# 测试代码 main 函数
def main():
    # vocab
    vocab_x, vocab_xr, vocab_y, vocab_yr = get_vocab()

    # x, y = encode_data(vocab_x, vocab_y) 
    # print(f"x: \n{x} \nx len: {len(x)}")
    # print(f"y: \n{y} \ny len: {len(y)}")
    
    # words_x, words_y = decode_data(x, y, vocab_xr, vocab_yr)
    # print(f"words_x: \n{words_x}")
    # print(f"words_y: \n{words_y}")

    # data loader
    train_dl, valid_dl = create_dataloader()
    for src, target in train_dl:
        print(f"src: \n{src} \nsrc shape: {src.size()}")
        print(f"target: \n{target} \ntarget shape: {target.size()}")
        break

if __name__ == "__main__":
    main()
