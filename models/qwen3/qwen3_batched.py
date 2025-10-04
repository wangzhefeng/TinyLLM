# -*- coding: utf-8 -*-

# ***************************************************
# * File        : qwen3_batched.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-10-02
# * Version     : 1.0.100217
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn

from layers.transformer_block import TransformerBlockQwen3_batched
from layers.normailzation.rms_norm import RMSNorm_Qwen3
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
            [TransformerBlockQwen3_batched(cfg) for _ in range(cfg.n_layers)]
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
        # self.current_pos = 0  # Track current position in KV cache
    
    def forward(self, x, cache=None, attn_mask=None):
        # token embedding layer
        tok_embeds = self.tok_embed(x)
        x = tok_embeds
        # tokenized text shape
        batch_size, num_tokens = x.shape[0], x.shape[1]
        # Derive pos_start from cache content (layer 0 K length) if present
        if cache is not None and cache.get(0) is not None:
            prev_k0, _ = cache.get(0)    # (batch_size, G_kv, L_prev, D)
            pos_start = prev_k0.size(2)  # L_prev
        else:
            pos_start = 0  # Not strictly necessary but helps torch.compile
        pos_end = pos_start + num_tokens
        # Build causal mask for [Q=num_tokens, K=pos_end]
        base = torch.triu(
            torch.ones(pos_end, pos_end, device=x.device, dtype=torch.bool), 
            diagonal=1,
        )
        causal4d = base[pos_start:pos_end, :pos_end][None, None, :, :]
        has_pad = attn_mask is not None and (~attn_mask[:, :pos_end]).any().item()
        if has_pad:
            # Mask out padded keys so they don't appear in the softmax denominator
            kpm = (attn_mask[:, :pos_end] == 0).view(batch_size, 1, 1, pos_end)
            mask = causal4d | kpm
        else:
            mask = causal4d
        pos_ids_current = torch.arange(pos_start, pos_end, device=x.device).unsqueeze(0).expand(batch_size, -1)
        # zero-out padded query rows so their Q/K/V become zeros and don't affect cache
        if attn_mask is not None:
            qmask = attn_mask[:, pos_start:pos_end].unsqueeze(-1)
            x = x * qmask.to(x.dtype)
        # transformer block
        for i, block in enumerate(self.trf_blocks):
            if cache is not None:
                block_cache = cache.get(i)
            else:
                block_cache = None
            x, new_block_cache = block(
                x, mask, self.cos, self.sin, 
                cache=block_cache, 
                pos_id=pos_ids_current
            )
            if cache is not None:
                cache.update(i, new_block_cache)
        # final RMSNorm
        x = self.final_norm(x)
        # linear output layer(head)
        logits = self.out_head(x.to(self.cfg.dtype))

        return logits
    
    def reset_kv_cache(self):
        """
        Keep for compatibility with regular, non-batched generate_text_basic_cache function
        """
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
