# -*- coding: utf-8 -*-

# ***************************************************
# * File        : phi15.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2026-10-24
# * Version     : 1.0.102421
# * Description : Phi-1.5（microsoft/phi-1_5，1.3B）教学实现：
# *               MHA + partial RoPE（只旋转每头前 partial_rotary_factor*head_dim
# *               维，其余直通）+ LayerNorm + gelu_new FFN，GPT-2 式串行残差。
# *               官方 config：d=2048、24 层、32 头、d_ff=8192、ctx=2048、
# *               vocab=51200、partial_rotary_factor=0.5、tie=false、
# *               layer_norm_eps=1e-5、hidden_act=gelu_new。
# *               checkpoint 的合并 Wqkv 投影由 models/io/phi 按行切片注入
# *               拆分式的 W_query/W_key/W_value。
# * Link        : https://huggingface.co/microsoft/phi-1_5
# * Requirement : torch
# ***************************************************

__all__ = ["Model", "PhiAttention", "PhiMLP", "PhiBlock"]

# python libraries
from pathlib import Path

import torch
import torch.nn as nn

from layers.normalization.layer_norm import LayerNorm
from layers.position_encoding.RoPE import precompute_rope_params, compute_rope

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


class PhiAttention(nn.Module):
    """MHA + partial RoPE：位置旋转只作用于每头前 ``rotary_dim`` 维，其余直通。"""

    def __init__(self, cfg):
        super().__init__()

        d_model = cfg.embed_dim
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim if cfg.head_dim is not None else d_model // self.n_heads
        self.d_out = self.n_heads * self.head_dim
        # partial rotary：phi-1.5 官方 partial_rotary_factor=0.5（每头前 32/64 维旋转）
        factor = getattr(cfg, "partial_rotary_factor", 1.0)
        self.rotary_dim = max(int(factor * self.head_dim), 2)
        # 拆分式 Q/K/V 投影（bias 对齐 HF Phi）
        self.W_query = nn.Linear(d_model, self.d_out, bias=True, dtype=cfg.dtype)
        self.W_key = nn.Linear(d_model, self.d_out, bias=True, dtype=cfg.dtype)
        self.W_value = nn.Linear(d_model, self.d_out, bias=True, dtype=cfg.dtype)
        self.out_proj = nn.Linear(self.d_out, d_model, bias=True, dtype=cfg.dtype)
        # 因果掩码（True=屏蔽）
        self.register_buffer(
            "mask",
            torch.triu(
                torch.ones(cfg.context_length, cfg.context_length, dtype=torch.bool),
                diagonal=1,
            ),
            persistent=False,
        )
        # RoPE 频率按 rotary_dim 计算
        cos, sin = precompute_rope_params(
            head_dim=self.rotary_dim,
            theta_base=cfg.rope_base,
            context_length=cfg.context_length,
        )
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, x):
        batch_size, num_tokens, _ = x.shape
        # (B, T, D) -> (B, H, T, hd)
        q = self.W_query(x).view(batch_size, num_tokens, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.W_key(x).view(batch_size, num_tokens, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.W_value(x).view(batch_size, num_tokens, self.n_heads, self.head_dim).transpose(1, 2)
        # partial RoPE：前 rotary_dim 维旋转，其余直通
        if self.rotary_dim < self.head_dim:
            q = torch.cat(
                [compute_rope(q[..., : self.rotary_dim], self.cos, self.sin),
                 q[..., self.rotary_dim:]], dim=-1)
            k = torch.cat(
                [compute_rope(k[..., : self.rotary_dim], self.cos, self.sin),
                 k[..., self.rotary_dim:]], dim=-1)
        else:
            q = compute_rope(q, self.cos, self.sin)
            k = compute_rope(k, self.cos, self.sin)
        # attention scores -> weights -> context
        attn_scores = q @ k.transpose(-2, -1) / self.head_dim**0.5
        attn_scores = attn_scores.masked_fill(self.mask[:num_tokens, :num_tokens], -torch.inf)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        context_vec = (attn_weights @ v).transpose(1, 2).contiguous().view(
            batch_size, num_tokens, self.d_out)
        return self.out_proj(context_vec)


class PhiMLP(nn.Module):
    """gelu_new（tanh 近似）FFN，带 bias（对齐 HF Phi 的 mlp.fc1/fc2）。"""

    def __init__(self, cfg):
        super().__init__()

        d_ff = cfg.d_ff if getattr(cfg, "d_ff", None) else cfg.hidden_dim
        self.fc1 = nn.Linear(cfg.embed_dim, d_ff, bias=True, dtype=cfg.dtype)
        self.fc2 = nn.Linear(d_ff, cfg.embed_dim, bias=True, dtype=cfg.dtype)
        self.gelu = nn.GELU(approximate="tanh")

    def forward(self, x):
        return self.fc2(self.gelu(self.fc1(x)))


class PhiBlock(nn.Module):
    """GPT-2 式串行残差：x = x + attn(ln1(x))；x = x + mlp(ln2(x))。"""

    def __init__(self, cfg):
        super().__init__()

        self.input_layernorm = LayerNorm(cfg.embed_dim, eps=1e-5)
        self.attn = PhiAttention(cfg)
        self.post_attention_layernorm = LayerNorm(cfg.embed_dim, eps=1e-5)
        self.ff = PhiMLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.input_layernorm(x))
        x = x + self.ff(self.post_attention_layernorm(x))
        return x


class Model(nn.Module):
    """Phi-1.5 decoder-only 模型（教学版，无 KV cache 路径）。"""

    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.tok_embed = nn.Embedding(cfg.vocab_size, cfg.embed_dim, dtype=cfg.dtype)
        self.blocks = nn.ModuleList([PhiBlock(cfg) for _ in range(cfg.n_layers)])
        self.final_norm = LayerNorm(cfg.embed_dim, eps=1e-5)
        self.out_head = nn.Linear(cfg.embed_dim, cfg.vocab_size, bias=False, dtype=cfg.dtype)

    def forward(self, x):
        h = self.tok_embed(x)
        for block in self.blocks:
            h = block(h)
        return self.out_head(self.final_norm(h))


# 测试代码 main 函数
def main():
    import torch

    from config.phi.model_cfgs import PHI15_TINY_CONFIG
    from utils.args_tools import DotDict
    from utils.log_util import logger

    torch.manual_seed(123)
    cfg = DotDict(PHI15_TINY_CONFIG)
    model = Model(cfg).eval()
    x = torch.randint(0, cfg.vocab_size, (2, 16))
    with torch.no_grad():
        logits = model(x)
    logger.info(f"forward: {tuple(logits.shape)}")
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"params: {total:,}")

if __name__ == "__main__":
    main()
