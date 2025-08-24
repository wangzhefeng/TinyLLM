# -*- coding: utf-8 -*-

# ***************************************************
# * File        : gpt.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-01-25
# * Version     : 1.0.012519
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

from layers.transformer_block import TransformerBlockGPT2_124M
from layers.normailzation.layer_norm import LayerNorm

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


class Model(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        # Embedding
        self.tok_embed = nn.Embedding(cfg.vocab_size, cfg.embed_dim)
        self.pos_embed = nn.Embedding(cfg.context_length, cfg.embed_dim)
        self.drop_embed = nn.Dropout(cfg.dropout)
        # transformer block
        self.trf_blocks = nn.Sequential(
            *[TransformerBlockGPT2_124M(cfg) for _ in range(cfg.n_layers)]
        )
        # LayerNorm
        self.final_norm = LayerNorm(cfg.embed_dim)
        # output head linear
        self.out_head = nn.Linear(cfg.embed_dim, cfg.vocab_size, bias=False)

    def forward(self, x):
        # [batch_size, num_tokens(seq_len), embed_dim]
        # TODO batch_size, seq_len, embed_dim = x.shape
        batch_size, seq_len = x.shape
        # token embedding, shape: [batch_size, num_tokens, embed_dim]
        tok_embeds = self.tok_embed(x)
        # position embedding, shape: [batch_size, num_tokens, embed_dim]
        pos_embeds = self.pos_embed(torch.arange(seq_len, device=x.device))
        # embedding, shape: [batch_size, num_tokens, embed_dim]
        x = tok_embeds + pos_embeds
        # dropout, shape: [batch_size, num_tokens, embed_dim]
        x = self.drop_embed(x)
        # transformer blocks, shape: [batch_size, num_tokens, embed_dim]
        x = self.trf_blocks(x)
        # final layer norm, shape: [batch_size, num_tokens, embed_dim]
        x = self.final_norm(x)
        # output head, shape: [batch_size, num_tokens, vocab_size]
        logits = self.out_head(x)

        return logits




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
