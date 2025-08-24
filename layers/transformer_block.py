# -*- coding: utf-8 -*-

# ***************************************************
# * File        : transformer_block.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-02-12
# * Version     : 1.0.021221
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

from layers.attention import (
    MultiHeadAttention,
    MultiHeadAttentionRoPE,
    GroupedQueryAttention,
)
from layers.feed_forward import FeedForwardGELU, FeedForwardSiLU
from layers.moe import SparseMoE
from layers.normailzation.layer_norm import LayerNorm
from layers.normailzation.rms_norm import RMSNorm

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


class TransformerBlockGPT2_124M(nn.Module):
    
    def __init__(self, cfg):
        super().__init__()
        
        self.attn = MultiHeadAttention(
            d_model = cfg.embed_dim,
            d_out = cfg.embed_dim,
            n_heads = cfg.n_heads,
            context_length = cfg.context_length,
            dropout = cfg.dropout,
            qkv_bias = cfg.qkv_bias,
        )
        self.ff = FeedForwardGELU(cfg)
        self.norm1 = LayerNorm(cfg.embed_dim)
        self.norm2 = LayerNorm(cfg.embed_dim)
        self.drop_shortcut = nn.Dropout(cfg.dropout)
    
    def forward(self, x):
        # shortcut connection for attention block
        shortcut = x
        x = self.norm1(x)  # shape: [batch_size, num_tokens, embed_dim]
        x = self.attn(x)  # shape: [batch_size, num_tokens, embed_dim]
        x = self.drop_shortcut(x)  # shape: [batch_size, num_tokens, embed_dim]
        x = shortcut + x  # shape: [batch_size, num_tokens, embed_dim]
        # shortcut connection for feed forward block
        shortcut = x
        x = self.norm2(x)  # shape: [batch_size, num_tokens, embed_dim]
        x = self.ff(x)  # shape: [batch_size, num_tokens, embed_dim]
        x = self.drop_shortcut(x)  # shape: [batch_size, num_tokens, embed_dim]
        x = shortcut + x  # shape: [batch_size, num_tokens, embed_dim]
        
        return x


class TransformerBlockMoE(nn.Module):
    """
    Mixture of Experts Transformer block
    communication followed by computation (multi-head self-attention + SparseMoE) 
    """

    def __init__(self, cfg):
        super().__init__()
        
        self.attn = MultiHeadAttention(
            d_model = cfg.embed_dim,
            d_out = cfg.embed_dim,
            n_heads = cfg.n_heads,
            context_length = cfg.context_length,
            dropout = cfg.dropout,
            qkv_bias = cfg.qkv_bias,
        )
        self.smoe = SparseMoE(cfg)
        self.norm1 = LayerNorm(cfg.embed_dim)
        self.norm2 = LayerNorm(cfg.embed_dim)
        self.drop_shortcut = nn.Dropout(cfg.dropout)
    
    def forward(self, x):
        # shortcut connection for attention block
        shortcut = x
        x = self.norm1(x)
        x = self.attn(x)  # shape: [batch_size, num_tokens, embed_dim]
        x = self.drop_shortcut(x)
        x = shortcut + x
        # shortcut connection for feed forward block
        shortcut = x
        x = self.norm2(x)
        x = self.smoe(x)
        x = self.drop_shortcut(x)
        x = shortcut + x
        
        return x


class TransformerBlockLlama2(nn.Module):
    
    def __init__(self, cfg):
        super().__init__()

        self.attn = MultiHeadAttentionRoPE(
            d_model = cfg.embed_dim,
            d_out = cfg.embed_dim,
            n_heads = cfg.n_heads,
            context_length=cfg.context_length,
            dtype = cfg.dtype,
        )
        self.ff = FeedForwardSiLU(cfg)
        self.norm1 = RMSNorm(cfg.embed_dim, eps=1e-5)
        self.norm2 = RMSNorm(cfg.embed_dim, eps=1e-5)
    
    def forward(self, x):
        # shortcut connection for attention block
        shortcut = x
        x = self.norm1(x)
        x = self.attn(x)  # shape: [batch_size, num_tokens, emb_size]
        x = x + shortcut
        # shortcut connection for feed forward block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = x + shortcut

        return x


class TransformerBlockLlama3(nn.Module):
    
    def __init__(self, cfg):
        super().__init__()

        self.attn = GroupedQueryAttention(
            d_model = cfg.embed_dim,
            d_out = cfg.embed_dim,
            n_heads = cfg.n_heads,
            context_length = cfg.context_length,
            num_kv_groups = cfg.n_kv_groups,
            rope_base = cfg.rope_base,
            rope_config = cfg.rope_freq,
            dtype = cfg.dtype,
        )
        self.ff = FeedForwardSiLU(cfg)
        self.norm1 = RMSNorm(cfg.embed_dim, eps=1e-5)
        self.norm2 = RMSNorm(cfg.embed_dim, eps=1e-5)
    
    def forward(self, x):
        # Shortcut connection for attention block
        shortcut = x
        x = self.norm1(x)
        x = self.attn(x.to(torch.bfloat16))  # Shape: [batch_size, num_tokens, emb_size]
        x = x + shortcut
        # Shortcut connection for feed-forward block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x.to(torch.bfloat16))
        x = x + shortcut

        return x




# 测试代码 main 函数
def main():
    import torch
    from utils.args_tools import DotDict
    from utils.log_util import logger

    # ------------------------------
    # Transformer Block test
    # ------------------------------
    # shape: [batch_size, num_tokens, embed_dim]
    GPT2_124M_CONFIG = {
        "vocab_size": 50257,
        "context_length": 1024,
        "embed_dim": 768,
        "n_heads": 12,
        "n_layers": 12,
        "dropout": 0.1,
        "qkv_bias": False,
        "dtype": torch.float32,
    }
    GPT2_124M_CONFIG = DotDict(GPT2_124M_CONFIG)

    # input
    torch.manual_seed(123)
    x = torch.rand(2, 4, 768)

    # transformer
    block = TransformerBlockGPT2_124M(GPT2_124M_CONFIG)
    output = block(x)
    logger.info(f"Input shape: {x.shape}")
    logger.info(f"Output shape: {output.shape}")

if __name__ == "__main__":
    main()
