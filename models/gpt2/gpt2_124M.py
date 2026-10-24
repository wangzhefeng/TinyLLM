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
from pathlib import Path

import torch
import torch.nn as nn

from layers.transformer_block import TransformerBlockGPT2_124M
from layers.normalization.layer_norm import LayerNorm

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
        # LayerNorm（自定义 scale/shift 版，与 inference/loaders/gpt2/ 的加载约定一致）
        self.final_norm = LayerNorm(cfg.embed_dim)
        # output head linear
        self.out_head = nn.Linear(cfg.embed_dim, cfg.vocab_size, bias=False)
        # weight tying：out_head 与 tok_embed 共享权重（GPT-2 官方权重两者同值）
        if getattr(cfg, "tie_word_embeddings", False):
            self.out_head.weight = self.tok_embed.weight

    def forward(self, x, use_cache=False, cache=None, attention_mask=None, position_ids=None,
                output_hidden_states=False):
        _, seq_len = x.shape
        external_cache = cache is not None
        if position_ids is not None:
            pos_id = position_ids
        else:
            if external_cache:
                pos_start = cache.position
            elif use_cache:
                pos_start = self.ptr_current_pos
            else:
                pos_start = 0
            if pos_start + seq_len > self.pos_embed.num_embeddings:
                raise IndexError("Position exceeds GPT-2 context length")
            pos_id = torch.arange(pos_start, pos_start + seq_len, device=x.device, dtype=torch.long)
        x = self.tok_embed(x) + self.pos_embed(pos_id).unsqueeze(0)
        x = self.drop_embed(x)
        for layer_idx, block in enumerate(self.trf_blocks):
            if external_cache:
                x, new_cache = block(
                    x,
                    cache=cache.get(layer_idx),
                    attention_mask=attention_mask,
                    return_cache=True,
                )
                cache.update(layer_idx, new_cache)
            else:
                x = block(x, use_cache=use_cache, attention_mask=attention_mask)
        if external_cache:
            cache.advance(seq_len)
        elif use_cache:
            self.ptr_current_pos += seq_len
        final_hidden = self.final_norm(x)
        if output_hidden_states:
            return self.out_head(final_hidden), final_hidden
        return self.out_head(final_hidden)
    
    def reset_kv_cache(self):
        for blk in self.trf_blocks:
            blk.attn.reset_cache()
        self.ptr_current_pos = 0




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
