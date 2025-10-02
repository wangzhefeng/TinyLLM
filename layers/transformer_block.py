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
    MHAPyTorchScaledDotProduct,
    MultiHeadAttentionRoPE,
    GroupedQueryAttention,
    GroupedQueryAttention_Qwen3,
    GroupedQueryAttention_Qwen3_optimized,
    GroupedQueryAttention_Qwen3_batched,
    GroupedQueryAttention_Gemma3,
)
from layers.feed_forward import (
    FeedForwardGELU, 
    FeedForwardSiLU,
    FeedForwardGELU_Gemma3, 
    MoEFeedForward,
)
from layers.moe import SparseMoE
# from layers.normailzation.layer_norm import LayerNorm
from layers.normailzation.rms_norm import (
    RMSNorm, 
    RMSNorm_Qwen3,
    RMSNorm_Gemma3,
)

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


class TransformerBlockGPT2_124M(nn.Module):
    
    def __init__(self, cfg):
        super().__init__()
        
        self.attn = MHAPyTorchScaledDotProduct(
            d_model = cfg.embed_dim,
            d_out = cfg.embed_dim,
            n_heads = cfg.n_heads,
            context_length = cfg.context_length,
            dropout = cfg.dropout,
            qkv_bias = cfg.qkv_bias,
            training=cfg.is_train,
            # window_size = cfg.kv_window_size if "kv_window_size" in cfg else cfg.context_length
        )
        self.ff = FeedForwardGELU(cfg)
        self.norm1 = nn.LayerNorm(cfg.embed_dim)
        self.norm2 = nn.LayerNorm(cfg.embed_dim)
        self.drop_shortcut = nn.Dropout(cfg.dropout)
    
    def forward(self, x, use_cache=False):
        # shortcut connection for attention block
        shortcut = x
        # LayerNorm 1
        x = self.norm1(x)          # shape: [batch_size, num_tokens, embed_dim]
        # Masked multi-head attention
        x = self.attn(
            x, 
            # use_cache=use_cache
        )  # shape: [batch_size, num_tokens, embed_dim]
        # Dropout
        x = self.drop_shortcut(x)  # shape: [batch_size, num_tokens, embed_dim]
        # Residual connection
        x = shortcut + x           # shape: [batch_size, num_tokens, embed_dim]
        # shortcut connection for feed forward block
        shortcut = x
        # LayerNorm 2
        x = self.norm2(x)          # shape: [batch_size, num_tokens, embed_dim]
        # Feed Forward
        x = self.ff(x)             # shape: [batch_size, num_tokens, embed_dim]
        # Dropout
        x = self.drop_shortcut(x)  # shape: [batch_size, num_tokens, embed_dim]
        # Residual connection
        x = shortcut + x           # shape: [batch_size, num_tokens, embed_dim]
        
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
        self.norm1 = nn.LayerNorm(cfg.embed_dim)
        self.norm2 = nn.LayerNorm(cfg.embed_dim)
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


class TransformerBlockQwen3(nn.Module):
    
    def __init__(self, cfg):
        super().__init__()

        self.attn = GroupedQueryAttention_Qwen3(
            d_model = cfg.embed_dim,
            n_heads = cfg.n_heads,
            num_kv_groups = cfg.n_kv_groups,
            head_dim = cfg.head_dim,
            qk_norm = cfg.qk_norm,
            dtype = cfg.dtype,
        )
        if cfg.num_experts > 0:
            self.ff = MoEFeedForward(cfg)
        else:
            self.ff = FeedForwardSiLU(cfg)
        self.norm1 = RMSNorm_Qwen3(cfg.embed_dim, eps=1e-6)
        self.norm2 = RMSNorm_Qwen3(cfg.embed_dim, eps=1e-6)
    
    def forward(self, x, mask, cos, sin, start_pos=0, cache=None):
        # Shortcut connection for attention block
        shortcut = x
        x = self.norm1(x)
        x, next_cache = self.attn(
            x.to(torch.bfloat16), mask, cos, sin, 
            start_pos=start_pos, 
            cache=cache, 
        )  # Shape: [batch_size, num_tokens, emb_size]
        x = x + shortcut

        # Shortcut connection for feed-forward block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x.to(torch.bfloat16))
        x = x + shortcut

        return x, next_cache


class TransformerBlockQwen3_optimized(nn.Module):
    
    def __init__(self, cfg):
        super().__init__()

        self.attn = GroupedQueryAttention_Qwen3_optimized(
            d_model = cfg.embed_dim,
            n_heads = cfg.n_heads,
            num_kv_groups = cfg.n_kv_groups,
            head_dim = cfg.head_dim,
            qk_norm = cfg.qk_norm,
            dtype = cfg.dtype,
        )
        if cfg.num_experts > 0:
            self.ff = MoEFeedForward(cfg)
        else:
            self.ff = FeedForwardSiLU(cfg)
        self.norm1 = RMSNorm_Qwen3(cfg.embed_dim, eps=1e-6)
        self.norm2 = RMSNorm_Qwen3(cfg.embed_dim, eps=1e-6)
    
    def forward(self, x, mask, cos, sin, start_pos=0, cache=None, layer_idx=None, exact=False):
        # Shortcut connection for attention block
        shortcut = x
        x = self.norm1(x)
        x = self.attn(
            x.to(torch.bfloat16), mask, cos, sin, 
            start_pos=start_pos, 
            cache=cache, 
            layer_idx=layer_idx, 
            exact=exact
        )  # Shape: [batch_size, num_tokens, emb_size]
        x = x + shortcut

        # Shortcut connection for feed-forward block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x.to(torch.bfloat16))
        x = x + shortcut

        return x


class TransformerBlockQwen3_batched(nn.Module):
    
    def __init__(self, cfg):
        super().__init__()

        self.attn = GroupedQueryAttention_Qwen3_batched(
            d_model = cfg.embed_dim,
            n_heads = cfg.n_heads,
            num_kv_groups = cfg.n_kv_groups,
            head_dim = cfg.head_dim,
            qk_norm = cfg.qk_norm,
            dtype = cfg.dtype,
        )
        if cfg.num_experts > 0:
            self.ff = MoEFeedForward(cfg)
        else:
            self.ff = FeedForwardSiLU(cfg)
        self.norm1 = RMSNorm_Qwen3(cfg.embed_dim, eps=1e-6)
        self.norm2 = RMSNorm_Qwen3(cfg.embed_dim, eps=1e-6)
    
    def forward(self, x, mask, cos, sin, cache=None, pos_ids=None):
        # Shortcut connection for attention block
        shortcut = x
        x = self.norm1(x)
        x, next_cache = self.attn(
            x.to(torch.bfloat16), mask, cos, sin, 
            cache=cache, pos_ids=pos_ids
        )  # Shape: [batch_size, num_tokens, emb_size]
        x = x + shortcut

        # Shortcut connection for feed-forward block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x.to(torch.bfloat16))
        x = x + shortcut

        return x, next_cache


class TransformerBlockGemma3(nn.Module):
    
    def __init__(self, cfg, attn_type):
        super().__init__()

        self.cfg = cfg
        self.attn_type = attn_type
        self.attn = GroupedQueryAttention_Gemma3(
            d_model = cfg.embed_dim,
            n_heads = cfg.n_heads,
            num_kv_groups = cfg.n_kv_groups,
            head_dim = cfg.head_dim,
            qk_norm = cfg.qk_norm,
            query_pre_attn_scalar=cfg.query_pre_attn_scalar,
            dtype = cfg.dtype,
        )
        self.ff = FeedForwardGELU_Gemma3(cfg)
        self.input_layernorm = RMSNorm_Gemma3(cfg.embed_dim, eps=1e-6)
        self.post_attention_layernorm = RMSNorm_Gemma3(cfg.embed_dim, eps=1e-6)
        self.pre_feedforward_layernorm = RMSNorm_Gemma3(cfg.embed_dim, eps=1e-6)
        self.post_feedforward_layernorm = RMSNorm_Gemma3(cfg.embed_dim, eps=1e-6)
    
    def forward(self, x, mask_global, mask_local, cos_global, sin_global, cos_local, sin_local):
        # Shortcut connection for attention block
        shortcut = x
        x = self.input_layernorm(x)

        if self.attn_type == "sliding_attention":
            attn_mask = mask_local
            cos = cos_local
            sin = sin_local
        else:
            attn_mask = mask_global
            cos = cos_global
            sin = sin_global
        # Shape: [batch_size, num_tokens, emb_size]
        x_attn = self.attn(x.to(torch.bfloat16), attn_mask, cos, sin)
        x_attn = self.post_attention_layernorm(x_attn)
        x = x_attn + shortcut
        # Shortcut connection for feed-forward block
        shortcut = x
        x_ffn = self.pre_feedforward_layernorm(x)
        x_ffn = self.ff(x_ffn)
        x_ffn = self.post_feedforward_layernorm(x_ffn)
        x = x_ffn + shortcut

        return x




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
