# -*- coding: utf-8 -*-

# ***************************************************
# * File        : feed_forward.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-02-12
# * Version     : 1.0.021221
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

import torch.nn as nn

from layers.activation import (
    ReLU, 
    ReLUPyTorch,
    GELU, 
    SiLU,
    SwiGLU,
)

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


class FeedForwardReLU(nn.Module):
    
    def __init__(self, cfg):
        super().__init__()

        self.fc1 = nn.Linear(cfg.embed_dim, cfg.d_ff, dtype=cfg.dtype, bias=True)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(cfg.d_ff, cfg.embed_dim, dtype=cfg.dtype, bias=True)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        out = self.fc2(x)

        return out


class FeedForwardGELU(nn.Module):
    
    def __init__(self, cfg):
        super().__init__()

        self.fc1 = nn.Linear(cfg.embed_dim, cfg.d_ff, dtype=cfg.dtype, bias=True)
        self.gelu = nn.GELU(approximate="tanh")
        self.fc2 = nn.Linear(cfg.d_ff, cfg.embed_dim, dtype=cfg.dtype, bias=True)
    
    def forward(self, x):
        # input tensor x.shape: [batch_size, num_tokens, embed_dim]
        # Linear layer
        x = self.fc1(x)  # [batch_size, num_tokens, 4*embed_dim]
        # GELU activation
        x = self.gelu(x)  # [batch_size, num_tokens, 4*embed_dim]
        # Linear layer
        out = self.fc2(x)  # [batch_size, num_tokens, embed_dim]

        return out


class FeedForwardSiLU(nn.Module):
    """
    SwiGLU: GLU Variants Improve Transformer (2020): https://arxiv.org/abs/2002.05202
    """
    def __init__(self, cfg):
        super().__init__()

        self.fc1 = nn.Linear(cfg.embed_dim, cfg.d_ff, dtype=cfg.dtype, bias=False)
        self.fc2 = nn.Linear(cfg.embed_dim, cfg.d_ff, dtype=cfg.dtype, bias=False)
        self.silu = nn.SiLU()
        self.fc3 = nn.Linear(cfg.d_ff, cfg.embed_dim, dtype=cfg.dtype, bias=False)

    def forward(self, x):
        x_fc1 = self.fc1(x)
        x_fc2 = self.fc2(x)
        x = self.silu(x_fc1) * x_fc2
        out = self.fc3(x)

        return out


class FeedForwardSwiGLU(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.swiglu = None

    def forward(self, x):
        pass




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
