# -*- coding: utf-8 -*-

# ***************************************************
# * File        : mla.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2026-10-20
# * Version     : 2.0.102020
# * Description : Multi-Head Latent Attention (MLA) 教学实现。
# *               MLA 最早由 DeepSeek-V2 提出，并被 V3/R1 沿用：
# *               把 K/V 压缩到低秩 latent，推断时只缓存 latent 与 decoupled
# *               RoPE key，再反投影出 K/V，显著降低 KV cache 显存。
# *               本实现忠实于论文（每步从 latent 反投影），不做 weight
# *               absorption（矩阵吸收）优化——教学上更清晰。
# * Link        : DeepSeek-V2 https://arxiv.org/abs/2405.04434
# *               DeepSeek-V3 https://arxiv.org/abs/2412.19437
# *               LLMs-from-scratch ch04/05_mla
# * Requirement : torch
# ***************************************************

__all__ = [
    "MultiHeadLatentAttention",
]

# python libraries
from pathlib import Path

import torch
import torch.nn as nn

from layers.normalization.rms_norm import RMSNorm
from layers.position_encoding.RoPE import precompute_rope_params, compute_rope

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


class MultiHeadLatentAttention(nn.Module):
    """Multi-Head Latent Attention（DeepSeek-V2/V3 式）。

    与 MHA/GQA 的核心区别（以 ``n_heads=32, kv_lora_rank=512`` 为例）：

    1. K/V 不是直接投影，而是先压到低秩 latent ``c_kv``（kv_lora_rank 维），
       推断时 **只缓存 c_kv + k_rope**（512 + 64 维），
       而 GQA 要缓存 2 × n_kv_heads × head_dim；MLA 显存开销与头数解耦。
    2. Decoupled RoPE：位置旋转只作用于每头的 rope 子维（qk_rope_head_dim），
       其余 nope 子维不受位置约束——否则 latent 反投影后 RoPE 无法解耦。

    Cache 协议：``cache = (c_kv, k_rope)``，
    ``c_kv: (B, T_total, kv_lora_rank)``，``k_rope: (B, H, T_total, qk_rope_head_dim)``。

    Args:
        d_model: 模型维度
        n_heads: 注意力头数
        kv_lora_rank: K/V latent 压缩维度（V3: 512）
        q_lora_rank: Query 压缩维度；None 表示不压缩 Q（V3: 1536）
        qk_nope_head_dim: 每头不受 RoPE 的子维（V3: 128）
        qk_rope_head_dim: 每头受 RoPE 的子维（V3: 64）
        v_head_dim: 每头 value 维（V3: 128）
        rope_base: RoPE theta
        context_length: 预计算 cos/sin 的长度
        dtype: 参数精度
    """

    def __init__(self, d_model: int, n_heads: int,
                 kv_lora_rank: int = 128, q_lora_rank: int | None = None,
                 qk_nope_head_dim: int = 64, qk_rope_head_dim: int = 32,
                 v_head_dim: int | None = None,
                 rope_base: float = 10_000.0, context_length: int = 4096,
                 dtype=None):
        super().__init__()

        v_head_dim = v_head_dim or qk_nope_head_dim
        self.n_heads = n_heads
        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.head_dim = qk_nope_head_dim + qk_rope_head_dim
        q_in = q_lora_rank if q_lora_rank is not None else d_model

        # ---- Query 路径（可选低秩压缩）----
        if q_lora_rank is not None:
            self.q_down_proj = nn.Linear(d_model, q_lora_rank, bias=False, dtype=dtype)
            self.q_layernorm = RMSNorm(q_lora_rank, eps=1e-6)
        else:
            self.q_down_proj = None
            self.q_layernorm = None
        # 上投影：latent -> 每头 nope 子维
        self.q_up_proj = nn.Linear(q_in, n_heads * qk_nope_head_dim, bias=False, dtype=dtype)
        # 每头 rope 子维直接从（压缩后的）q 表征投影
        self.q_rope_proj = nn.Linear(q_in, n_heads * qk_rope_head_dim, bias=False, dtype=dtype)

        # ---- K/V 路径（latent 压缩，推断只缓存它）----
        self.kv_down_proj = nn.Linear(d_model, kv_lora_rank, bias=False, dtype=dtype)
        self.kv_layernorm = RMSNorm(kv_lora_rank, eps=1e-6)
        # 从 latent 反投影出每头 nope key / value
        self.k_up_proj = nn.Linear(kv_lora_rank, n_heads * qk_nope_head_dim, bias=False, dtype=dtype)
        self.v_up_proj = nn.Linear(kv_lora_rank, n_heads * v_head_dim, bias=False, dtype=dtype)
        # decoupled RoPE key：与 latent 并行的独立投影
        self.k_rope_proj = nn.Linear(d_model, n_heads * qk_rope_head_dim, bias=False, dtype=dtype)

        # ---- 输出投影 ----
        self.o_proj = nn.Linear(n_heads * v_head_dim, d_model, bias=False, dtype=dtype)

        # RoPE 只作用于 rope 子维
        cos, sin = precompute_rope_params(
            head_dim=qk_rope_head_dim, theta_base=rope_base,
            context_length=context_length, dtype=dtype or torch.float32,
        )
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, x, start_pos: int = 0, cache=None):
        """前向。

        Args:
            x: (B, T, d_model)
            start_pos: 增量解码时新 token 的起始位置（RoPE offset / 因果 mask offset）
            cache: 上一层的 ``(c_kv, k_rope)``；None 表示全新序列

        Returns:
            (attn_out, next_cache)
        """
        batch_size, num_tokens, _ = x.shape
        device = x.device
        H, nope, rope, vdim = self.n_heads, self.qk_nope_head_dim, self.qk_rope_head_dim, self.v_head_dim

        # ---- Q ----
        q_in = x
        if self.q_down_proj is not None:
            q_in = self.q_layernorm(self.q_down_proj(x))
        q_nope = self.q_up_proj(q_in).view(batch_size, num_tokens, H, nope).transpose(1, 2)
        q_rope = self.q_rope_proj(q_in).view(batch_size, num_tokens, H, rope).transpose(1, 2)
        q_rope = compute_rope(q_rope, self.cos, self.sin, offset=start_pos)

        # ---- K/V latent（新 token）----
        c_kv_new = self.kv_layernorm(self.kv_down_proj(x))            # (B, T, kv_lora_rank)
        k_rope_new = self.k_rope_proj(x).view(batch_size, num_tokens, H, rope).transpose(1, 2)
        k_rope_new = compute_rope(k_rope_new, self.cos, self.sin, offset=start_pos)

        # ---- Cache：只存 latent + rope key（MLA 显存优势的来源）----
        if cache is not None:
            prev_c, prev_kr = cache
            c_kv = torch.cat([prev_c, c_kv_new], dim=1)               # (B, T_total, kv_lora_rank)
            k_rope = torch.cat([prev_kr, k_rope_new], dim=2)          # (B, H, T_total, rope)
        else:
            c_kv, k_rope = c_kv_new, k_rope_new
        next_cache = (c_kv, k_rope)

        # ---- 从 latent 反投影（教学实现：每步重投影；不做 weight absorption）----
        total_len = c_kv.shape[1]
        k_nope = self.k_up_proj(c_kv).view(batch_size, total_len, H, nope).transpose(1, 2)
        v = self.v_up_proj(c_kv).view(batch_size, total_len, H, vdim).transpose(1, 2)

        # ---- Attention：scores = q_nope·k_nope + q_rope·k_rope ----
        attn_scores = q_nope @ k_nope.transpose(2, 3) + q_rope @ k_rope.transpose(2, 3)
        attn_scores = attn_scores / (self.head_dim ** 0.5)

        # 因果 mask（带 cache offset）
        row_idx = torch.arange(start_pos, start_pos + num_tokens, device=device).unsqueeze(1)
        col_idx = torch.arange(total_len, device=device).unsqueeze(0)
        causal_mask = row_idx < col_idx                               # True = 屏蔽
        attn_scores = attn_scores.masked_fill(causal_mask, -torch.inf)

        attn_weights = torch.softmax(attn_scores, dim=-1)
        context_vec = (attn_weights @ v).transpose(1, 2).reshape(batch_size, num_tokens, H * vdim)

        return self.o_proj(context_vec), next_cache
