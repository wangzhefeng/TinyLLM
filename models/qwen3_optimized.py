# -*- coding: utf-8 -*-

# ***************************************************
# * File        : qwen3_06B.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-08-31
# * Version     : 1.0.083120
# * Description : Qwen3-0.6B
# *               Qwen3-0.6B-KVCache
# *               Qwen3-MoE
# *                 Qwen3-30B-A3B(Coder, Instruct, Thinking)
# *                 Qwen3-Coder-30B-A3B-Instruct
# *                 Qwen3 Coder Flash(30B-A3B Mixture of Experts)
# *               Qwen3-MoE-KVCache
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)

import torch
import torch.nn as nn

from layers.transformer_block import TransformerBlockQwen3_optimized
from layers.normailzation.rms_norm import RMSNorm_Qwen3
from layers.position_encoding.RoPE import precompute_rope_params

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


class Model(nn.Module):

    def __init__(self, cfg, exact=False):
        super().__init__()

        self.cfg = cfg
        # Embedding
        self.tok_embed = nn.Embedding(cfg.vocab_size, cfg.embed_dim, dtype=cfg.dtype)
        # transformer block
        self.trf_blocks = nn.ModuleList(
            [TransformerBlockQwen3_optimized(cfg) for _ in range(cfg.n_layers)]
        )
        # RMSNorm
        self.final_norm = RMSNorm_Qwen3(cfg.embed_dim)
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
        self.exact = exact
        self.current_pos = 0  # Track current position in KV cache
    
    def forward(self, x, cache=None):
        # tokenized text shape
        batch_size, num_tokens = x.shape
        # token embedding layer
        tok_embeds = self.tok_embed(x)
        x = tok_embeds
        # kv cache
        if cache is not None:
            pos_start = self.current_pos
            pos_end = pos_start + num_tokens
            self.current_pos = pos_end 
            mask = torch.triu(
                torch.full((num_tokens, num_tokens), -torch.inf, device=x.device, dtype=self.cfg.dtype),
                diagonal=1 + pos_start,
            )
        else:
            pos_start = 0  # Not strictly necessary but helps torch.compile
            mask = torch.triu(
                torch.full((num_tokens, num_tokens), -torch.inf, device=x.device, dtype=self.cfg.dtype),
                diagonal=1,
            )
        # Shape (1, 1, num_tokens, num_tokens) to broadcast across batch and heads
        mask = mask[None, None, :, :]
        # transformer block
        for i, block in enumerate(self.trf_blocks):
            if cache is not None:
                cache.allocate(i, x.size(0))
            x = block(
                x, mask, self.cos, self.sin, 
                start_pos=pos_start, 
                cache=cache, 
                layer_idx=i,
                exact=self.exact,
            )
        # final RMSNorm
        x = self.final_norm(x)
        # linear output layer(head)
        logits = self.out_head(x.to(self.cfg.dtype))

        return logits
    
    def reset_kv_cache(self):
        self.current_pos = 0




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()

