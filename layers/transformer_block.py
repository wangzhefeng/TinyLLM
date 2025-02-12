# -*- coding: utf-8 -*-

# ***************************************************
# * File        : transformer_block.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-02-12
# * Version     : 1.0.021221
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# * TODO        : 1.
# ***************************************************

__all__ = []

# python libraries
import os
import sys
ROOT = str(os.getcwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)

import torch.nn as nn

from layers.attention import MultiHeadAttention
from layers.feed_forward import FeedForward
from layers.layer_norm import LayerNorm

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


class TransformerBlock(nn.Module):
    
    def __init__(self, cfg):
        super().__init__()
        
        self.attn = MultiHeadAttention(
            d_in = cfg["emb_dim"],
            d_out = cfg["emb_dim"],
            context_length = cfg["context_length"],
            num_heads = cfg["n_heads"],
            dropout = cfg["drop_rate"],
            qkv_bias = cfg["qkv_bias"],
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])
    
    def forward(self, x):
        # shortcut connection for attention block
        shortcut = x
        x = self.norm1(x)
        x = self.attn(x)
        x = self.drop_shortcut(x)
        x = shortcut + x
        # shortcut connection for feed forward block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = shortcut + x
        
        return x






# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
