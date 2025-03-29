# -*- coding: utf-8 -*-

# ***************************************************
# * File        : ffn.py
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
from layers.activation import GELU, SiLU

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


class FeedForward(nn.Module):
    
    def __init__(self, cfg):
        super(FeedForward, self).__init__()

        self.fc1 = nn.Linear(cfg.emb_dim, 4 * cfg.emb_dim, dtype=cfg.dtype, bias=True)
        self.fc2 = nn.Linear(4 * cfg.emb_dim, cfg.emb_dim, dtype=cfg.dtype, bias=True)
        self.silu = GELU()
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.silu(x)
        out = self.fc2(x)

        return out


class FeedForwardSiLU(nn.Module):
    """
    SwiGLU: GLU Variants Improve Transformer (2020): https://arxiv.org/abs/2002.05202
    """
    
    def __init__(self, cfgs):
        super(FeedForwardSiLU, self).__init__()

        self.fc1 = nn.Linear(cfgs.emb_dim, cfgs.hidden_dim, dtype=cfgs.dtype, bias=False)
        self.fc2 = nn.Linear(cfgs.emb_dim, cfgs.hidden_dim, dtype=cfgs.dtype, bias=False)
        self.fc3 = nn.Linear(cfgs.hidden_dim, cfgs.emb_dim, dtype=cfgs.dtype, bias=False)
        self.silu = SiLU()

    def forward(self, x):
        x_fc1 = self.fc1(x)
        x_fc2 = self.fc2(x)
        x = self.silu(x_fc1) * x_fc2
        out = self.fc3(x)

        return out




# 测试代码 main 函数
def main():
    import torch
    from utils.args_tools import DotDict
    from utils.log_util import logger
    # ------------------------------
    # Feed Forward test
    # ------------------------------
    # params
    GPT_CONFIG_124M = {
        "vocab_size": 50257,
        "context_length": 1024,
        "emb_dim": 768,
        "n_heads": 12,
        "n_layers": 12,
        "hidden_dim": 100,
        "dtype": torch.float32,
        "dropout": 0.1,
        "qkv_bias": False,
    }
    GPT_CONFIG_124M = DotDict(GPT_CONFIG_124M)

    # feed forward
    ffn = FeedForward(GPT_CONFIG_124M)
    ffn_silu = FeedForwardSiLU(GPT_CONFIG_124M)
    
    # forward
    x = torch.rand(2, 3, 768)
    out = ffn(x)
    out_silu = ffn_silu(x)
    
    logger.info(f"out: {out}")
    logger.info(f"out_silu: {out_silu}")
    logger.info(f"out.shape: {out.shape}")
    logger.info(f"out_silu.shape: {out_silu.shape}")

if __name__ == "__main__":
    main()
