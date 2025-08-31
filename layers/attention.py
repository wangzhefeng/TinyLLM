# -*- coding: utf-8 -*-

# ***************************************************
# * File        : attention.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-01-25
# * Version     : 1.0.012513
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
from packaging.version import parse as parse_version

import torch
import torch.nn as nn

from layers.position_encoding.RoPE import (
    precompute_rope_params, 
    compute_rope
)

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


class CasualAttention(nn.Module):
    """
    Casual Self Attention
    """
    def __init__(self, d_model: int, d_out: int, context_length: int, dropout: float=0.0, qkv_bias: bool=False):
        super().__init__()

        self.W_query = nn.Linear(d_model, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_model, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_model, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))
    
    def forward(self, x):
        batch_size, num_tokens, embed_dim = x.shape
        # q, k, v
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)
        # attention scores
        attn_scores = queries @ keys.transpose(1, 2)
        # casual mask
        attn_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        # attention weights
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        # attention weights dropout
        attn_weights = self.dropout(attn_weights)
        # context scores
        context_vec = attn_weights @ values
        
        return context_vec


class MultiHeadAttentionWrapper(nn.Module):
    
    def __init__(self, d_model: int, d_out: int, n_heads: int, context_length: int, dropout: float=0.0, qkv_bias: bool=False, proj: bool=False):
        super().__init__()
        
        self.heads = nn.ModuleList([
            CasualAttention(d_model, d_out // n_heads, context_length, dropout, qkv_bias) 
            for _ in range(n_heads)
        ])
        self.proj = proj
        if proj:
            self.out_proj = nn.Linear(d_out, d_out)

    def forward(self, x):
        x = torch.cat([head(x) for head in self.heads], dim=-1)
        if self.proj:
            x = self.out_proj(x)

        return x


class MultiHeadAttention(nn.Module):
    """
    GPT2, small, medium, large, XL
    """
    def __init__(self, d_model: int, d_out: int, n_heads: int, 
                 context_length: int, dropout: float=0.0, qkv_bias: bool=False, 
                 max_seq_len: int=None, window_size: int=None):
        super().__init__()

        assert (d_out % n_heads == 0), "d_out must be divisible by n_heads"
        
        self.d_out = d_out
        self.n_heads = n_heads
        self.head_dim = d_out // n_heads
        # query, key, value weights
        self.W_query = nn.Linear(d_model, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_model, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_model, d_out, bias=qkv_bias)
        # Linear layer to combine head outputs
        self.out_proj = nn.Linear(d_out, d_out)
        # dropout
        self.dropout = nn.Dropout(dropout)
        # mask buffer
        self.max_seq_len = max_seq_len or context_length
        self.window_size = window_size or self.max_seq_len
        # self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1).bool(), persistent=False)
        self.register_buffer("cache_k", None, persistent=False)
        self.register_buffer("cache_v", None, persistent=False)
        # self.ptr_current_pos = 0
    
    def forward(self, x, use_cache=False):
        # shape: [batch_size, num_tokens, d_model]
        batch_size, num_tokens, embed_dim = x.shape
        # Query, Key, Value
        # shape: [batch_size, num_tokens, d_model]->[batch_size, num_tokens, d_out]
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)
        # Split the matrix by adding a "n_heads" dimension, unroll last dim
        # shape: [batch_size, num_tokens, d_out]->[batch_size, num_tokens, n_heads, head_dim]
        queries = queries.view(batch_size, num_tokens, self.n_heads, self.head_dim)
        keys =       keys.view(batch_size, num_tokens, self.n_heads, self.head_dim)
        values =   values.view(batch_size, num_tokens, self.n_heads, self.head_dim) 
        # Transpose
        # shape: [batch_size, num_tokens, n_heads, head_dim]->[batch_size, n_heads, num_tokens, head_dim]
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        # KV-Cache
        if use_cache:
            if self.cache_k is None or self.cache_k.size(0) != batch_size:
                self.cache_k = torch.zeros(batch_size, self.n_heads, self.window_size, self.head_dim, device=x.device)
                self.cache_v = torch.zeros_like(self.cache_k)
                self.ptr_cur = 0  # pointer to next free slot
            
            # if incoming chunk would overflow discard oldest tokens
            if self.ptr_cur + num_tokens > self.window_size:
                overflow = self.ptr_cur + num_tokens - self.window_size
                # shift everything left by 'overflow' (cheap view-copy)
                self.cache_k[:, :, :-overflow, :] = self.cache_k[:, :, overflow:, :].clone()
                self.cache_v[:, :, :-overflow, :] = self.cache_v[:, :, overflow:, :].clone()
                self.ptr_cur -= overflow  # pointer after shift
            
            self.cache_k[:, :, self.ptr_cur:(self.ptr_cur + num_tokens), :] = keys
            self.cache_v[:, :, self.ptr_cur:(self.ptr_cur + num_tokens), :] = values
            self.ptr_cur += num_tokens

            keys = self.cache_k[:, :, :self.ptr_cur, :]
            values = self.cache_v[:, :, :self.ptr_cur, :]
        else:
            keys, values = keys, values
            self.ptr_cur = 0  # keep pointer sane if you interleave modes
        # ------------------------------
        # scaled dot-product attention(aka self-attention) with a causal mask
        # ------------------------------
        # dot product for each head
        # [_, _, num_tokens, head_dim]->[_, _, head_dim, num_tokens]->[_, _, num_tokens, num_tokens]
        attn_scores = queries @ keys.transpose(2, 3)

        # mask to fill attention scores, [batch_size, n_heads, num_tokens, num_tokens] 
        # num_tokens_Q = queries.shape[-2]
        # num_tokens_K = keys.shape[-2]
        # if use_cache:
        #     mask_bool = self.mask[self.ptr_current_pos:(self.ptr_current_pos + num_tokens_Q), :num_tokens_K]
        #     self.ptr_current_pos += num_tokens_Q
        # else:
        #     mask_bool = self.mask[:num_tokens_Q, :num_tokens_K]
        
        K = attn_scores.size(-1)
        if num_tokens == K:
            # No cache → use the pre‑baked triangular mask slice
            causal_mask = torch.triu(torch.ones(num_tokens, K, device=x.device, dtype=torch.bool), diagonal=1)
        else:
            # Cached: need to offset the diagonal by (K − num_tokens)
            offset = K - num_tokens  # number of tokens already in cache before this chunk
            row_idx = torch.arange(num_tokens, device=x.device).unsqueeze(1)  # (num_tokens, 1)
            col_idx = torch.arange(K, device=x.device).unsqueeze(0)           # (1, K)
            causal_mask = row_idx + offset < col_idx                          # True where j > i+offset
        
        # use the mask to fill attention scores, [batch_size, n_heads, num_tokens, num_tokens]
        attn_scores.masked_fill_(causal_mask.unsqueeze(0).unsqueeze(0), -torch.inf)
        # ------------------------------
        # attention weights
        # ------------------------------
        # [batch_size, n_heads, num_tokens, num_tokens]
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)
        # ------------------------------
        # context vector
        # ------------------------------
        # shape:[_, _, num_tokens, num_tokens]@[_, _, num_tokens, head_dim]->[batch_size, num_tokens, n_heads, head_dim]
        context_vec = (attn_weights @ values).transpose(1, 2)
        # combine heads, where self.d_out = self.n_heads * self.head_dim
        context_vec = context_vec.contiguous().view(batch_size, num_tokens, self.d_out)
        # optional projection
        context_vec = self.out_proj(context_vec)

        return context_vec
    
    def reset_cache(self):
        self.cache_k, self.cache_v = None, None


class MultiHeadAttentionCombinedQKV(nn.Module):
    
    def __init__(self, d_model: int, d_out: int, n_heads: int, context_length: int, dropout: float=0.0, qkv_bias: bool=False):
        super().__init__()

        assert (d_out % n_heads == 0), "d_out must be divisible by n_heads"

        self.n_heads = n_heads
        self.context_length = context_length
        self.head_dim = d_out // n_heads
        # query, key, value weights
        self.qkv = nn.Linear(d_model, 3 * d_out, bias=qkv_bias)
        # Linear layer to combine head outputs
        self.out_proj = nn.Linear(d_out, d_out)
        # dropout
        self.dropout = nn.Dropout(dropout)
        # mask buffer
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1).bool())

    def forward(self, x):
        # shape: [batch_size, num_tokens, d_model]
        batch_size, num_tokens, embed_dim = x.shape
        # ------------------------------
        # Query, Key, Value
        # ------------------------------
        # (batch_size, num_tokens, embed_dim)->(batch_size, num_tokens, 3 * embed_dim)
        qkv = self.qkv(x)
        # (batch_size, num_tokens, 3 * embed_dim)->(batch_size, num_tokens, 3, n_heads, head_dim)
        qkv = qkv.view(batch_size, num_tokens, 3, self.n_heads, self.head_dim)
        # (batch_size, num_tokens, 3, n_heads, head_dim)->(3, b, n_heads, num_tokens, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        # (3, b, n_heads, num_tokens, head_dim)->3 times (batch_size, num_head, num_tokens, head_dim)
        queries, keys, values = qkv.unbind(0)
        # ------------------------------
        # attention scores
        # ------------------------------
        # (batch_size, n_heads, num_tokens, head_dim)->(batch_size, n_heads, num_tokens, num_tokens)
        attn_scores = queries @ keys.transpose(-2, -1)
        attn_scores = attn_scores.masked_fill(self.mask[:num_tokens, :num_tokens], -torch.inf)
        # ------------------------------
        # TODO attention weights
        # ------------------------------
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**-0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)
        # ------------------------------
        # context vectors
        # ------------------------------
        # (batch_size, n_heads, num_tokens, num_tokens)->(batch_size, n_heads, num_tokens, head_dim)
        context_vec = attn_weights @ values
        # (batch_size, n_heads, num_tokens, head_dim)->(batch_size, num_tokens, n_heads, head_dim)
        context_vec = context_vec.transpose(1, 2)
        # (batch_size, num_tokens, n_heads, head_dim)->(batch_size, num_tokens, embed_dim)
        context_vec = context_vec.contiguous().view(batch_size, num_tokens, embed_dim)
        context_vec = self.out_proj(context_vec)

        return context_vec


class MHAEinsum(nn.Module):

    def __init__(self, d_model: int, d_out: int, n_heads: int, context_length: int, dropout: float=0.0, qkv_bias: bool=False):
        super().__init__()

        assert d_out % n_heads == 0, "d_out must be divisible by n_heads"

        self.d_out = d_out
        self.n_heads = n_heads
        self.head_dim = d_out // n_heads
        # Initialize parameters for Q, K, V
        self.W_query = nn.Parameter(torch.randn(d_out, d_model))
        self.W_key = nn.Parameter(torch.randn(d_out, d_model))
        self.W_value = nn.Parameter(torch.randn(d_out, d_model))

        if qkv_bias:
            self.bias_q = nn.Parameter(torch.zeros(d_out))
            self.bias_k = nn.Parameter(torch.zeros(d_out))
            self.bias_v = nn.Parameter(torch.zeros(d_out))
        else:
            self.register_parameter("bias_q", None)
            self.register_parameter("bias_k", None)
            self.register_parameter("bias_v", None)

        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

        # Initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.W_query, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W_key, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W_value, a=math.sqrt(5))
        if self.bias_q is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.W_query)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias_q, -bound, bound)
            nn.init.uniform_(self.bias_k, -bound, bound)
            nn.init.uniform_(self.bias_v, -bound, bound)

    def forward(self, x):
        batch_size, num_tokens, embed_dim = x.shape

        # Calculate Q, K, V using einsum, first perform linear transformations
        Q = torch.einsum("bnd,di->bni", x, self.W_query)
        K = torch.einsum("bnd,di->bni", x, self.W_key)
        V = torch.einsum("bnd,di->bni", x, self.W_value)

        # Add biases if they are used
        if self.bias_q is not None:
            Q += self.bias_q
            K += self.bias_k
            V += self.bias_v

        # Reshape for multi-head attention
        Q = Q.view(batch_size, num_tokens, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, num_tokens, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, num_tokens, self.n_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.einsum("bhnd,bhmd->bhnm", Q, K) / (self.head_dim ** 0.5)

        # Apply mask
        mask = self.mask[:num_tokens, :num_tokens].unsqueeze(0).unsqueeze(1).expand(batch_size, self.n_heads, num_tokens, num_tokens)
        scores = scores.masked_fill(mask.bool(), -torch.inf)

        # Softmax and dropout
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Aggregate the attended context vectors
        context_vec = torch.einsum("bhnm,bhmd->bhnd", attn_weights, V)

        # Combine heads and project the output
        context_vec = context_vec.transpose(1, 2).reshape(batch_size, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)

        return context_vec


class MHAPyTorchScaledDotProduct(nn.Module):
    
    def __init__(self, d_model: int, d_out: int, n_heads: int, context_length: int, dropout: float=0.0, qkv_bias: bool=False, training: bool=False):
        super().__init__()

        assert d_out % n_heads == 0, "embed_dim is indivisible by n_heads"

        self.d_out = d_out
        self.n_heads = n_heads
        self.head_dim = d_out // n_heads
        self.context_length = context_length

        self.qkv = nn.Linear(d_model, 3 * d_out, bias=qkv_bias)
        self.proj = nn.Linear(d_out, d_out)
        self.dropout = dropout
        self.training = training

    def forward(self, x):
        batch_size, num_tokens, embed_dim = x.shape
        # (batch_size, num_tokens, embed_dim)->(batch_size, num_tokens, 3 * embed_dim)
        qkv = self.qkv(x)
        # (batch_size, num_tokens, 3 * embed_dim)->(batch_size, num_tokens, 3, n_heads, head_dim)
        qkv = qkv.view(batch_size, num_tokens, 3, self.n_heads, self.head_dim)
        # (batch_size, num_tokens, 3, n_heads, head_dim)->(3, b, n_heads, num_tokens, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        # (3, b, n_heads, num_tokens, head_dim)->3 times (batch_size, n_heads, num_tokens, head_dim)
        queries, keys, values = qkv

        use_dropout = 0.0 if not self.training else self.dropout
        context_vec = nn.functional.scaled_dot_product_attention(
            queries, keys, values, 
            attn_mask=None, 
            dropout_p=use_dropout, 
            is_causal=True
        )
        # Combine heads, where self.d_out = self.n_heads * self.head_dim
        context_vec = context_vec.transpose(1, 2).contiguous().view(batch_size, num_tokens, self.d_out)
        context_vec = self.proj(context_vec)

        return context_vec


class MHAPyTorchSDPAWithoutFlash(nn.Module):
    
    def __init__(self, d_model: int, d_out: int, n_heads: int, context_length: int, dropout: float=0.0, qkv_bias: bool=False, training: bool=False):
        super().__init__()

        assert d_out % n_heads == 0, "embed_dim is indivisible by n_heads"

        self.d_out = d_out
        self.n_heads = n_heads
        self.head_dim = d_out // n_heads
        self.context_length = context_length

        self.qkv = nn.Linear(d_model, 3 * d_out, bias=qkv_bias)
        self.proj = nn.Linear(d_out, d_out)
        self.dropout = dropout
        self.training = training
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1).bool())

    def forward(self, x):
        batch_size, num_tokens, embed_dim = x.shape
        # (batch_size, num_tokens, embed_dim)->(batch_size, num_tokens, 3 * embed_dim)
        qkv = self.qkv(x)
        # (batch_size, num_tokens, 3 * embed_dim)->(batch_size, num_tokens, 3, n_heads, head_dim)
        qkv = qkv.view(batch_size, num_tokens, 3, self.n_heads, self.head_dim)
        # (batch_size, num_tokens, 3, n_heads, head_dim)->(3, b, n_heads, num_tokens, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        # (3, b, n_heads, num_tokens, head_dim)->3 times (batch_size, n_heads, num_tokens, head_dim)
        queries, keys, values = qkv

        use_dropout = 0.0 if not self.training else self.dropout
        # Ensure attn_mask is compatible with expected shape and `batch_first=True`
        # No need to manually adjust for n_heads; ensure it's right for the sequence
        if self.context_length >= num_tokens:
            attn_mask = self.mask[:num_tokens, :num_tokens]
        else:
            attn_mask = self.mask[:self.context_length, :self.context_length]

        context_vec = nn.functional.scaled_dot_product_attention(
            queries, keys, values, 
            attn_mask=attn_mask, 
            dropout_p=use_dropout, 
            is_causal=False
        )
        # Combine heads, where self.d_out = self.n_heads * self.head_dim
        context_vec = context_vec.transpose(1, 2).contiguous().view(batch_size, num_tokens, self.d_out)
        context_vec = self.proj(context_vec)

        return context_vec


class MHAPyTorchClass(nn.Module):
    
    def __init__(self, d_model: int, d_out: int, n_heads: int, context_length: int, dropout: float=0.0, qkv_bias=False, need_weights=True):
        super().__init__()
        
        assert (d_out % n_heads == 0), "d_out must be divisible by n_heads"

        self.context_length = context_length
        self.need_weights = need_weights
        # MHA
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=d_out,
            num_heads=n_heads,
            dropout=dropout,
            bias=qkv_bias,
            add_bias_kv=qkv_bias,
            batch_first=True,
        )
        # Linear layer to combine head outputs
        self.out_proj = nn.Linear(d_out, d_out)
        # mask buffer
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1).bool())

    def forward(self, x):
        # shape: [batch_size, num_tokens, d_model]
        batch_size, num_tokens, embed_dim = x.shape
        # Ensure attn_mask is compatible with expected shape and `batch_first=True`
        # No need to manually adjust for n_heads; ensure it's right for the sequence
        if self.context_length >= num_tokens:
            attn_mask = self.mask[:num_tokens, :num_tokens]
        else:
            attn_mask = self.mask[:self.context_length, :self.context_length]
        # attn_mask broadcasting will handle batch_size dimension implicitly
        attn_output, _ = self.multihead_attn(
            x, x, x, 
            attn_mask=attn_mask, 
            need_weights=self.need_weights
        )
        # Linear layer to combine head outputs
        output = self.out_proj(attn_output)

        return output


class MHAPyTorchFlexAttention(nn.Module):

    def __init__(self, d_model: int, d_out: int, n_heads: int, context_length: int, dropout: float=0.0, qkv_bias: bool=False, training: bool=False):
        super().__init__()

        assert d_out % n_heads == 0, "embed_dim is indivisible by n_heads"

        self.d_out = d_out
        self.n_heads = n_heads
        self.head_dim = d_out // n_heads
        self.context_length = context_length

        self.qkv = nn.Linear(d_model, 3 * d_out, bias=qkv_bias)
        self.proj = nn.Linear(d_out, d_out)
        self.dropout = dropout
        self.training = training
        # self.register_buffer("block_mask", create_block_mask(causal, B=None, H=None, Q_LEN=context_length, KV_LEN=context_length))
        # `create_block_mask` function does not support buffers, yet
        self.current_version = self.normalize_version(torch.__version__)
        MIN_TORCH_VERSION = "2.5.0"
        self.required_version = parse_version(MIN_TORCH_VERSION)
        if self.current_version >= self.required_version:
            from torch.nn.attention.flex_attention import flex_attention, create_block_mask
        self.block_mask = create_block_mask(self.causal, B=None, H=None, Q_LEN=context_length, KV_LEN=context_length)

    @staticmethod
    def normalize_version(version):
        parsed_version = parse_version(version)
        return parse_version(f"{parsed_version.major}.{parsed_version.minor}.{parsed_version.micro}")

    @staticmethod
    def causal(batch_size, h, q_idx, kv_idx):
        return q_idx >= kv_idx

    def forward(self, x):
        batch_size, num_tokens, embed_dim = x.shape
        # (batch_size, num_tokens, embed_dim)->(batch_size, num_tokens, 3 * embed_dim)
        qkv = self.qkv(x)
        # (batch_size, num_tokens, 3 * embed_dim)->(batch_size, num_tokens, 3, n_heads, head_dim)
        qkv = qkv.view(batch_size, num_tokens, 3, self.n_heads, self.head_dim)
        # (batch_size, num_tokens, 3, n_heads, head_dim)->(3, b, n_heads, num_tokens, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        # (3, b, n_heads, num_tokens, head_dim)->3 times (batch_size, n_heads, num_tokens, head_dim)
        queries, keys, values = qkv
        # ------------------------------
        # mask
        # ------------------------------
        use_dropout = 0.0 if not self.training else self.dropout
        # Ensure attn_mask is compatible with expected shape and `batch_first=True`
        # No need to manually adjust for n_heads; ensure it's right for the sequence
        if self.context_length >= num_tokens:
            attn_mask = self.block_mask[:num_tokens, :num_tokens]
        else:
            attn_mask = self.block_mask[:self.context_length, :self.context_length]
        # ------------------------------
        # context vector
        # ------------------------------
        if self.current_version >= self.required_version:
            from torch.nn.attention.flex_attention import flex_attention, create_block_mask
        context_vec = flex_attention(queries, keys, values, block_mask=attn_mask)
        # Combine heads, where self.d_out = self.n_heads * self.head_dim
        context_vec = context_vec.transpose(1, 2).contiguous().view(batch_size, num_tokens, self.d_out)
        context_vec = self.proj(context_vec)

        return context_vec


class MultiHeadAttentionRoPE(nn.Module):
    
    def __init__(self, d_model: int, d_out: int, n_heads: int, context_length: int, dtype = None):
        super().__init__()

        assert (d_out % n_heads == 0), "d_out must be divisible by n_heads"
        
        self.d_out = d_out
        self.n_heads = n_heads
        # reduce the projection dim to match desired output dim
        self.head_dim = d_out // n_heads
        # query, key, value weights
        self.W_query = nn.Linear(d_model, d_out, bias=False, dtype=dtype)
        self.W_key = nn.Linear(d_model, d_out, bias=False, dtype=dtype)
        self.W_value = nn.Linear(d_model, d_out, bias=False, dtype=dtype)
        # Linear layer to combine head outputs
        self.out_proj = nn.Linear(d_out, d_out, bias=False, dtype=dtype)
        # mask
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1).bool())
        # RoPE
        cos, sin = precompute_rope_params(head_dim=self.head_dim, context_length=context_length)
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)
    
    def forward(self, x):
        # shape: [batch_size, num_tokens, d_model]
        batch_size, num_tokens, embed_dim = x.shape
        # ------------------------------
        # Query, Key, Value
        # ------------------------------
        # shape: [batch_size, num_tokens, d_out]
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)
        # split the matrix by adding a "n_heads" dimension, unroll last dim
        # shape: (batch_size, num_tokens, d_out)->(batch_size, num_tokens, n_heads, head_dim)
        queries = queries.view(batch_size, num_tokens, self.n_heads, self.head_dim)
        keys = keys.view(batch_size, num_tokens, self.n_heads, self.head_dim)
        values = values.view(batch_size, num_tokens, self.n_heads, self.head_dim)
        # transpose
        # shape: (batch_size, num_tokens, n_heads, head_dim)->(batch_size, n_heads, num_tokens, head_dim)
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        # RoPE
        queries = compute_rope(queries, self.cos, self.sin)
        keys = compute_rope(keys, self.cos, self.sin)
        # ------------------------------
        # scaled dot-product attention(aka self-attention) with a causal mask
        # ------------------------------
        # dot product for each head
        attn_scores = queries @ keys.transpose(2, 3)
        # mask to fill attention scores
        mask_bool = self.mask[:num_tokens, :num_tokens]
        # use the mask to fill attention scores
        attn_scores.masked_fill_(mask_bool, -torch.inf)
        # ------------------------------
        # attention weights
        # ------------------------------
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        # ------------------------------
        # context vector
        # ------------------------------
        # shape: (batch_size, num_tokens, n_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2)
        # combine heads, where self.d_out = self.n_heads * self.head_dim
        context_vec = context_vec.contiguous().view(batch_size, num_tokens, self.d_out)
        # or
        # context_vec = context_vec.reshape(batch_size, num_tokens, self.d_out)
        # optional projection
        context_vec = self.out_proj(context_vec)

        return context_vec


class SharedBuffers:
    _buffers = {}

    @staticmethod
    def get_buffers(context_length, head_dim, rope_base, freq_config, dtype=torch.float32):
        key = (
            context_length, 
            head_dim, 
            rope_base, 
            tuple(freq_config.values()) if freq_config else freq_config, 
            dtype,
        )
        if key not in SharedBuffers._buffers:
            # Create or fetch the buffers
            mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
            cos, sin = precompute_rope_params(head_dim, rope_base, context_length, freq_config)
            if dtype is not None:
                cos = cos.to(dtype)
                sin = sin.to(dtype)
            SharedBuffers._buffers[key] = (mask, cos, sin)

        return SharedBuffers._buffers[key]


class GroupedQueryAttention(nn.Module):
    
    def __init__(self, d_model: int, d_out: int, n_heads: int, context_length: int, num_kv_groups: int, 
                 rope_base: int=10_000, rope_config=None, dtype=None):
        super().__init__()

        assert d_out % n_heads == 0, "d_out must be divisible by n_heads"
        assert n_heads % num_kv_groups == 0, "n_heads must be divisible by num_kv_groups"  # NEW

        self.d_out = d_out
        self.n_heads = n_heads
        self.head_dim = d_out // n_heads
        # query, key, value weights
        # self.W_key = nn.Linear(d_model, d_out, bias=False, dtype=dtype)
        # self.W_value = nn.Linear(d_model, d_out, bias=False, dtype=dtype)
        self.W_key = nn.Linear(d_model, num_kv_groups * self.head_dim, bias=False, dtype=dtype)
        self.W_value = nn.Linear(d_model, num_kv_groups * self.head_dim, bias=False, dtype=dtype)
        self.num_kv_groups = num_kv_groups
        self.group_size = n_heads // num_kv_groups
        self.W_query = nn.Linear(d_model, d_out, bias=False, dtype=dtype)
        # out projection
        self.out_proj = nn.Linear(d_out, d_out, bias=False, dtype=dtype)
        # Fetch buffers using SharedBuffers
        mask, cos, sin = SharedBuffers.get_buffers(context_length, self.head_dim, rope_base, rope_config, dtype)
        self.register_buffer("mask", mask)
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)

    def forward(self, x):
        batch_size, num_tokens, embed_dim = x.shape

        queries = self.W_query(x)  # Shape: (batch_size, num_tokens, d_out)
        keys = self.W_key(x)  # Shape: (batch_size, num_tokens, num_kv_groups * head_dim)
        values = self.W_value(x)  # Shape: (batch_size, num_tokens, num_kv_groups * head_dim)

        # Reshape queries, keys, and values
        queries = queries.view(batch_size, num_tokens, self.n_heads, self.head_dim)
        # keys = keys.view(batch_size, num_tokens, self.n_heads, self.head_dim)
        # values = values.view(batch_size, num_tokens, self.n_heads, self.head_dim)
        keys = keys.view(batch_size, num_tokens, self.num_kv_groups, self.head_dim)
        values = values.view(batch_size, num_tokens, self.num_kv_groups, self.head_dim)
        # Transpose keys, values, and queries
        keys = keys.transpose(1, 2)  # Shape: (batch_size, n_heads, num_tokens, head_dim)
        values = values.transpose(1, 2)  # Shape: (batch_size, n_heads, num_tokens, head_dim)
        queries = queries.transpose(1, 2)  # Shape: (batch_size, num_query_groups, num_tokens, head_dim)
        # Apply RoPE
        keys = compute_rope(keys, self.cos, self.sin)
        queries = compute_rope(queries, self.cos, self.sin)

        # Expand keys and values to match the number of heads
        # Shape: (batch_size, n_heads, num_tokens, head_dim)
        keys = keys.repeat_interleave(self.group_size, dim=1)  # Shape: (batch_size, n_heads, num_tokens, head_dim)
        values = values.repeat_interleave(self.group_size, dim=1)  # Shape: (batch_size, n_heads, num_tokens, head_dim)
        # For example, before repeat_interleave along dim=1 (query groups):
        #   [K1, K2]
        # After repeat_interleave (each query group is repeated group_size times):
        #   [K1, K1, K2, K2]
        # If we used regular repeat instead of repeat_interleave, we'd get:
        #   [K1, K2, K1, K2]

        # Compute scaled dot-product attention (aka self-attention) with a causal mask
        # Shape: (batch_size, n_heads, num_tokens, num_tokens)
        attn_scores = queries @ keys.transpose(2, 3)  # Dot product for each head

        # Original mask truncated to the number of tokens and converted to boolean
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        # Use the mask to fill attention scores
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        assert keys.shape[-1] == self.head_dim

        # Shape: (batch_size, num_tokens, n_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2)

        # Combine heads, where self.d_out = self.n_heads * self.head_dim
        context_vec = context_vec.reshape(batch_size, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)  # optional projection

        return context_vec




# 测试代码 main 函数
def main():
    x = torch.randn(2, 6, 2, 3)
    print(x)
    print(x.shape)
    x = x.contiguous()
    print(x)
    print(x.shape)
    x = x.view(2, 6, 6)
    print(x)
    print(x.shape)

if __name__ == "__main__":
    main()
