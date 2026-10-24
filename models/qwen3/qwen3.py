# -*- coding: utf-8 -*-

# ***************************************************
# * File        : qwen3.py
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
from pathlib import Path

import torch
import torch.nn as nn

from layers.transformer_block import TransformerBlockQwen3
from layers.normalization.rms_norm import RMSNorm_Qwen3
from layers.position_encoding.RoPE import precompute_rope_params

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


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
        self.current_pos = 0  # Track current position in KV cache
    
    def forward(self, x, cache=None, attention_mask=None, position_ids=None):
        _, num_tokens = x.shape
        if position_ids is not None:
            pos_start = int(position_ids[0, 0].item())
        elif cache is not None:
            pos_start = cache.position
        else:
            pos_start = 0
        pos_end = pos_start + num_tokens
        if pos_end > self.cfg.context_length:
            raise IndexError(
                f"Position {pos_end} exceeds context_length {self.cfg.context_length}"
            )
        key_len = pos_end if cache is not None else num_tokens
        row = torch.arange(num_tokens, device=x.device).unsqueeze(1)
        col = torch.arange(key_len, device=x.device).unsqueeze(0)
        mask = (row + pos_start < col)[None, None, :, :]
        if attention_mask is not None:
            if attention_mask.ndim == 2:
                attention_mask = attention_mask[:, None, None, :]
            mask = mask | attention_mask
        x = self.tok_embed(x)
        for i, block in enumerate(self.trf_blocks):
            block_cache = cache.get(i) if cache is not None else None
            x, new_block_cache = block(
                x, mask, self.cos, self.sin,
                start_pos=pos_start,
                cache=block_cache,
            )
            if cache is not None:
                cache.update(i, new_block_cache)
        if cache is not None:
            cache.advance(num_tokens)
        return self.out_head(self.final_norm(x).to(self.out_head.weight.dtype))
    
    def reset_kv_cache(self):
        self.current_pos = 0




# 测试代码 main 函数
def main():
    import torch

    from utils.device import device_setting
    from utils.model_memory import model_memory_size
    from config.qwen3.model_cfgs import get_cfgs
    from utils.log_util import logger

    # random seed
    torch.manual_seed(123)

    # device
    device = device_setting(verbose=True)

    # model
    QWEN3_CONFIG = get_cfgs(CHOOSE_MODEL="0.6B")
    model = Model(cfg=QWEN3_CONFIG)
    model.to(device)
    logger.info(f"model: \n{model}") 

    # model forward
    input_tensor = torch.tensor([1, 2, 3]).unsqueeze(0)
    input_tensor = input_tensor.to(device)
    logger.info(f"input_tensor: {input_tensor}")
    logger.info(f"input_tensor shape: {input_tensor.shape}")

    output_tensor = model(input_tensor)
    logger.info(f"output_tensor: \n{output_tensor}")
    logger.info(f"output_tensor shape: {output_tensor.shape}")

    model_memory_size(model, input_dtype=torch.float32, verbose=True)
    model_memory_size(model, input_dtype=torch.bfloat16, verbose=True)

if __name__ == "__main__":
    main()
