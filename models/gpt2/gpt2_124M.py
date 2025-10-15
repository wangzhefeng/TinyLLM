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

__all__ = [
    "Model",
]

# python libraries
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)

import torch
import torch.nn as nn

from layers.transformer_block import TransformerBlockGPT2_124M
# from layers.normailzation.layer_norm import LayerNorm

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
        self.trf_blocks = nn.ModuleList(
            [TransformerBlockGPT2_124M(cfg) for _ in range(cfg.n_layers)]
        )
        self.ptr_current_pos = 0
        # LayerNorm
        self.final_norm = nn.LayerNorm(cfg.embed_dim)
        # output head linear
        self.out_head = nn.Linear(cfg.embed_dim, cfg.vocab_size, bias=False)

    def forward(self, x, use_cache=False):
        # tokenized text shape
        batch_size, seq_len = x.shape                    # shape: [batch_size, num_tokens(seq_len)]
        # token embedding layer
        tok_embeds = self.tok_embed(x)                   # shape: [batch_size, num_tokens, embed_dim]
        # positional embedding layer
        if use_cache:
            pos_id = torch.arange(
                self.ptr_current_pos, 
                self.ptr_current_pos + seq_len, 
                device=x.device, 
                dtype=torch.long
            )
            self.ptr_current_pos += seq_len
        else:
            pos_id = torch.arange(
                0, 
                seq_len, 
                device=x.device, 
                dtype=torch.long
            )                                            # shape: [batch_size, num_tokens, embed_dim]
        pos_embeds = self.pos_embed(pos_id).unsqueeze(0) # shape: [batch_size, num_tokens, embed_dim]
        # embedding
        x = tok_embeds + pos_embeds                      # shape: [batch_size, num_tokens, embed_dim]
        # dropout
        x = self.drop_embed(x)                           # shape: [batch_size, num_tokens, embed_dim]
        # transformer blocks
        for block in self.trf_blocks:
            x = block(x, use_cache=use_cache)            # shape: [batch_size, num_tokens, embed_dim]
        # final layer norm
        x = self.final_norm(x)                           # shape: [batch_size, num_tokens, embed_dim]
        # linear output layer(head)
        logits = self.out_head(x)                        # shape: [batch_size, num_tokens, vocab_size]

        return logits
    
    def reset_kv_cache(self):
        for blk in self.trf_blocks:
            blk.attn.reset_cache()
        self.ptr_current_pos = 0




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
