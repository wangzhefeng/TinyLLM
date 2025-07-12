# -*- coding: utf-8 -*-

# ***************************************************
# * File        : transformer_encdec.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-07-13
# * Version     : 1.0.071302
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
import math
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn

from layers.attention import (
    MultiHeadAttention,
    MultiHeadAttentionRoPE,
    GroupedQueryAttention,
)
from layers.feed_forward import (
    FeedForwardReLU,
    FeedForwardGELU,
    FeedForwardSiLU,
)
from layers.rms_norm import RMSNorm
from layers.layer_norm import LayerNorm
from layers.RoPE import precompute_rope_params, compute_rope
from layers.WoEmbed import WoEmbed
from layers.FixPE import PositionalEncoding

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


ffn_type = {
    "relu": FeedForwardReLU,
    "gelu": FeedForwardGELU,
    "silu": FeedForwardSiLU,
}
mha_type = {
    "mha": MultiHeadAttention,
    "mha_rope": MultiHeadAttentionRoPE,
    "gqa": GroupedQueryAttention,
}
norm_type = {
    "rms": RMSNorm,
    "ln": LayerNorm,
    "ln_torch": nn.LayerNorm
}
woe_type = {
    "torch": nn.Embedding,
    "others": None,
}
wpe_type = {
    "fixed": PositionalEncoding,
    "rope": compute_rope,
}


class EncoderLayer(nn.Module):

    def __init__(self, 
                 ffn: str, mha: str, norm: str,
                 d_model: int, num_heads: int, d_ff: int, dropout: float=0.1):
        super().__init__()

        self.self_attn = mha_type[mha](d_model, num_heads)
        self.feed_forward = ffn_type[ffn](d_model, d_ff)
        self.norm1 = norm_type[norm](d_model)
        self.norm2 = norm_type[norm](d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # self-attention with residual connection and layer normalization
        attn_output = self.self_attn(x, x, x, mask)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)
        # feed forward with residual connection and layer normalization
        ff_output = self.feed_forward(x)
        x = x + self.dropout(ff_output)
        x = self.norm2(x)

        return x


class Encoder(nn.Module):
    
    def __init__(self, woe: str, wpe: str,
                 vocab_size: int, d_model: int, num_heads: int, 
                 d_ff: int, num_layers: int, dropout: float=0.1):
        super().__init__()

        # params
        self.d_model = d_model
        # layers
        self.word_embedding = woe_type[woe](vocab_size, d_model)
        self.positional_encoding = wpe_type[wpe](d_model)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # embedding and positional encoding
        x = self.word_embedding(x) * math.sqrt(self.d_model)
        x = x + self.positional_encoding(x)
        x = self.dropout(x)
        # pass through encoder layers
        for layer in self.layers:
            x = layer(x, mask)
        
        return x


class DecoderLayer(nn.Module):
    
    def __init__(self, 
                 ffn: str, mha: str, norm: str,
                 d_model: int, num_heads: int, d_ff: int, dropout: float=0.1):
        super().__init__()

        self.self_attn = mha_type[mha](d_model, num_heads)
        self.cross_attn = mha_type[mha](d_model, num_heads)
        self.feed_forward = ffn_type[ffn](d_model, d_ff)
        self.norm1 = norm_type[norm](d_model)
        self.norm2 = norm_type[norm](d_model)
        self.norm3 = norm_type[norm](d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        # self-attention with residual connection and layer normalization
        self_attn_output = self.self_attn(x, x, x, tgt_mask)
        x = x + self.dropout(self_attn_output)
        x = self.norm1(x)
        # cross-attention with residual connection and layer normalization
        cross_attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = x + self.dropout(cross_attn_output)
        x = self.norm2(x)
        # feed forward with residual connection and layer normalization
        ff_output = self.feed_forward(x)
        x = x + self.dropout(ff_output)
        x = self.norm3(x)

        return x


class Decoder(nn.Module):
    
    def __init__(self, woe: str, wpe: str,
                 vocab_size: int, d_model: int, num_heads: int, 
                 d_ff: int, num_layers: int, dropout: float=0.1):
        super().__init__()

        # params
        self.d_model = d_model
        # layers
        self.word_embedding = woe_type[woe](vocab_size, d_model)
        self.positional_encoding = wpe_type[wpe](d_model)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        # embedding and positional encoding
        x = self.word_embedding(x) * math.sqrt(self.d_model)
        x = x + self.positional_encoding(x)
        x = self.dropout(x)
        # pass through decoder layers
        for layer in self.layers:
            x = layer(x, enc_output, src_mask, tgt_mask)
        
        return x




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
