# -*- coding: utf-8 -*-

# ***************************************************
# * File        : qwen3_06B.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-08-31
# * Version     : 1.0.083120
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

__all__ = [
    "Model"
]

# python libraries
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)

import torch
import torch.nn as nn

from layers.transformer_block import TransformerBlockQwen3
from layers.normailzation.rms_norm import RMSNorm_Qwen
from layers.position_encoding.RoPE import precompute_rope_params

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


# Control base model and the reasoning("thinking") model flag
USE_REASONING_MODEL = True


class Model(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        # Embedding
        self.tok_embed = nn.Embedding(cfg.vocab_size, cfg.embed_dim, dtype=cfg.dtype)
        # transformer block
        self.trf_blocks = nn.ModuleList(
            [TransformerBlockQwen3(cfg) for _ in range(cfg.n_layers)]
        )
        # RMSNorm
        self.final_norm = RMSNorm_Qwen(cfg.embed_dim)
        # output head linear
        self.out_head = nn.Linear(cfg.embed_dim, cfg.vocab_size, bias=False, dtype=cfg.dtype)
        # head dim
        if cfg.head_dim is None:
            head_dim = cfg.embed_dim // cfg.n_heads
        else:
            head_dim = cfg.head_dim
        # RoPE params
        cos, sin = precompute_rope_params(
            head_dim = head_dim,
            theta_base = cfg.rope_base,
            context_length = cfg.context_length,
        )
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
    
    def forward(self, x):
        # tokenized text shape
        # batch_size, seq_len, embed_dim = x.shape
        batch_size, seq_len = x.shape
        # token embedding layer
        tok_embeds = self.tok_embed(x)
        x = tok_embeds
        # mask
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), diagonal=1)
        # transformer block
        for block in self.trf_blocks:
            x = block(x, mask, self.cos, self.sin)
        # final RMSNorm
        x = self.final_norm(x)
        # linear output layer(head)
        logits = self.out_head(x.to(self.cfg.dtype))

        return logits




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
