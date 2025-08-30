# -*- coding: utf-8 -*-

# ***************************************************
# * File        : layer_norm.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-02-12
# * Version     : 1.0.021221
# * Description : description
# * Link        : paper: https://arxiv.org/abs/1607.06450
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

__all__ = [
    "LayerNorm",
]

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


class LayerNorm(nn.Module):
    """
    Layer Normalization
    
    Formular: `\gamma \times (x - \mu) / \sqrt{\sigma^2 + \epsilon} + \beta`
    """

    def __init__(self, embed_dim: int, eps: float = 1e-5):
        super().__init__()
 
        # gamma(trainable parameters)
        self.scale = nn.Parameter(torch.ones(embed_dim))
        # beta(trainable parameters)
        self.shift = nn.Parameter(torch.zeros(embed_dim))
        # a small constant for numerical stability(typically 1e-6)
        self.eps = eps

    def forward(self, x):
        # mu
        mean = x.mean(dim=-1, keepdim=True)
        # power of sigma
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        # normalization
        norm_x = (x - mean) / torch.sqrt(var + self.eps)

        return self.scale * norm_x + self.shift


class LayerNormPyTorch(nn.Module):
    """
    Layer Normalization
    
    Formular: `\gamma \times (x - \mu) / \sqrt{\sigma^2 + \epsilon} + \beta`
    """

    def __init__(self, embed_dim: int, eps: float = 1e-5):
        super().__init__() 

        self.ln = nn.LayerNorm(embed_dim, eps=eps)

    def forward(self, x):
        x = self.ln(x)

        return x




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
