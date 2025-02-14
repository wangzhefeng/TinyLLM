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
from utils.activation import GELU

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


class FeedForward(nn.Module):
    
    def __init__(self, cfg):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(cfg.emb_dim, 4 * cfg.emb_dim),
            GELU(),
            nn.Linear(4 * cfg.emb_dim, cfg.emb_dim),
        )
    
    def forward(self, x):
        return self.layers(x)




# 测试代码 main 函数
def main():
    import torch
    from utils.log_util import logger
    # ------------------------------
    # Feed Forward test
    # ------------------------------
    GPT_CONFIG_124M = {
        "vocab_size": 50257,
        "context_length": 1024,
        "emb_dim": 768,
        "n_heads": 12,
        "n_layers": 12,
        "dropout": 0.1,
        "qkv_bias": False,
    }
    ffn = FeedForward(GPT_CONFIG_124M)
    x = torch.rand(2, 3, 768)
    out = ffn(x)
    logger.info(f"out: {out}")
    logger.info(f"out.shape: {out.shape}")

if __name__ == "__main__":
    main()
