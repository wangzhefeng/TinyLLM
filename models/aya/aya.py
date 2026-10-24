# -*- coding: utf-8 -*-

# ***************************************************
# * File        : aya.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2026-10-24
# * Version     : 1.0.102421
# * Description : Tiny Aya（Cohere，260M）教学实现，Cohere Command
# *               架构家族风格。
# *
# *               ⚠ 架构未经官方 config 核实：cohere-for-ai/tinyaya 为
# *               门控模型（HTTP 401），config.json 无法拉取；以下实现基于
# *               任务描述的公开特性（GQA + CAT 复合仿射位置编码 + uvqk
# *               注意力偏置 + weight tying + SwiGLU + RMSNorm）与该家族
# *               的通用设计完成，细节（uvqk 的具体组合方式、CAT 的仿射
# *               参数化、超参数）为教学版合理假设，以零初始化保持
# *               "初始即恒等/近似恒等"。
# *
# *               关键组件：
# *               1. ComplexAffinePositionEncoding（CAT，自包含于本文件）：
# *                  head_dim 相邻两维配对视为复数 z = x[2k] + i·x[2k+1]，
# *                  施加位置相关旋转 exp(i·θ_t)（θ 频率与 RoPE 同型），
# *                  再做每维可学习仿射 (1+s)·y + b；s/b 零初始化 →
# *                  初始为纯旋转（位置 0 恒等，任意位置保范数、内积只依赖
# *                  相对位置）；
# *               2. GroupedQueryAttentionAya：GQA + CAT 作用于 q/k +
# *                  uvqk 四组每头每维可学习偏置（零初始化，以向量注入方式
# *                  分别作用于 q/k/v/注意力输出，避免被 softmax 消去）；
# *               3. Model：pre-norm RMSNorm + SwiGLU FFN + tied out_head。
# * Link        : https://huggingface.co/cohere-for-ai/tinyaya
# * Requirement : torch
# ***************************************************

__all__ = [
    "ComplexAffinePositionEncoding",
    "GroupedQueryAttentionAya",
    "TransformerBlockAya",
    "Model",
]

# python libraries
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn

from layers.attention import _expand_grouped_kv
from layers.feed_forward import FeedForwardSiLU
from layers.normalization.rms_norm import RMSNorm

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


class ComplexAffinePositionEncoding(nn.Module):
    """
    CAT（Complex Affine Transformation）位置编码。

    把 head_dim 的相邻两维 (x[2k], x[2k+1]) 视为一个复数
    z_k = x[2k] + i·x[2k+1]，对位置 t 施加复旋转

        z_k(t) = z_k * exp(i·θ_{t,k})，θ_{t,k} = t / theta_base^(2k/head_dim)

    再做每维可学习仿射 y = x_rot * (1 + scale) + shift。

    scale/shift 零初始化：初始时模块退化为纯复旋转——位置 0 为恒等映射，
    任意位置保范数，且 ⟨q(t), k(t+s)⟩ 只依赖相对位置 s（复乘为酉变换）。
    """

    def __init__(self, head_dim, theta_base=10_000.0, max_seq_len=4_096,
                 dtype=torch.float32):
        super().__init__()

        assert head_dim % 2 == 0, "head_dim must be even"
        self.head_dim = head_dim
        # 每个复数对的逆频率：inv_freq_k = theta_base^(-2k/head_dim)
        inv_freq = 1.0 / (
            theta_base ** (torch.arange(0, head_dim, 2, dtype=dtype) / head_dim)
        )
        positions = torch.arange(max_seq_len, dtype=dtype)
        angles = torch.outer(positions, inv_freq)  # (max_seq_len, head_dim//2)
        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)
        # 每维可学习缩放/平移（零初始化 → 初始为纯旋转）
        self.scale = nn.Parameter(torch.zeros(head_dim))
        self.shift = nn.Parameter(torch.zeros(head_dim))

    def forward(self, x, offset=0):
        """
        Args:
            x: (batch_size, n_heads, seq_len, head_dim)
            offset: 序列起始位置（增量推断用）

        Returns:
            同形状的旋转 + 仿射结果
        """
        batch_size, n_heads, seq_len, head_dim = x.shape
        assert head_dim == self.head_dim

        cos = self.cos[offset:(offset + seq_len)].to(x.dtype).unsqueeze(0).unsqueeze(0)
        sin = self.sin[offset:(offset + seq_len)].to(x.dtype).unsqueeze(0).unsqueeze(0)
        # 相邻维配对成复数：实部/虚部
        x_re = x[..., 0::2]  # (b, h, T, D//2)
        x_im = x[..., 1::2]
        # (x_re + i·x_im) * (cos + i·sin)
        y_re = x_re * cos - x_im * sin
        y_im = x_re * sin + x_im * cos
        y = torch.stack((y_re, y_im), dim=-1).flatten(-2)  # (b, h, T, D)
        # 每维可学习仿射（零初始化时为恒等）
        return y * (1 + self.scale).to(y.dtype) + self.shift.to(y.dtype)


class GroupedQueryAttentionAya(nn.Module):
    """
    Tiny Aya 风格 GQA：CAT 位置编码 + uvqk 注意力偏置。

    uvqk 偏置（官方细节未核实，本实现的教学解读）：四组每头每维可学习
    偏置向量，全部零初始化，分别注入注意力内部的四路张量——

    - q_bias / k_bias：投影 reshape 后、CAT 旋转前，加到 query / key 的
      每头每维上（作为每头的内容先验，参与后续点积）；
    - v_bias：加到 value 的每头每维上；
    - u_bias：加到每头注意力输出（context vector）的每维上。

    实现为向量注入而非纯加性 logit 偏置的原因：只依赖 query 的加性
    logit 项对每行是常数，会被 softmax 整行消去；注入 q/k 向量则通过
    点积同时依赖 query 与 key，四组偏置均有效。零初始化时本模块精确
    退化为标准缩放点积 GQA。
    """

    def __init__(self, d_model, n_heads, num_kv_groups, head_dim=None,
                 theta_base=10_000.0, context_length=4_096, dtype=None):
        super().__init__()

        assert n_heads % num_kv_groups == 0, "n_heads must be divisible by num_kv_groups"
        if head_dim is None:
            assert d_model % n_heads == 0, "`d_model` must be divisible by `n_heads` if `head_dim` is not set"
            head_dim = d_model // n_heads

        self.n_heads = n_heads
        self.num_kv_groups = num_kv_groups
        self.head_dim = head_dim
        self.d_out = n_heads * head_dim
        self.group_size = n_heads // num_kv_groups
        self.scaling = head_dim ** -0.5
        # query, key, value weights
        self.W_query = nn.Linear(d_model, self.d_out, bias=False, dtype=dtype)
        self.W_key = nn.Linear(d_model, num_kv_groups * head_dim, bias=False, dtype=dtype)
        self.W_value = nn.Linear(d_model, num_kv_groups * head_dim, bias=False, dtype=dtype)
        # out projection
        self.out_proj = nn.Linear(self.d_out, d_model, bias=False, dtype=dtype)
        # CAT 位置编码（每层独立持有，仿射参数逐层学习）
        self.cat = ComplexAffinePositionEncoding(
            head_dim, theta_base=theta_base, max_seq_len=context_length,
        )
        # uvqk 注意力偏置：四组每头每维，零初始化
        self.q_bias = nn.Parameter(torch.zeros(n_heads, head_dim))
        self.k_bias = nn.Parameter(torch.zeros(n_heads, head_dim))
        self.u_bias = nn.Parameter(torch.zeros(n_heads, head_dim))
        self.v_bias = nn.Parameter(torch.zeros(n_heads, head_dim))

    def forward(self, x, mask):
        batch_size, num_tokens, embed_dim = x.shape
        # Apply projections
        queries = self.W_query(x)   # (b, T, n_heads * head_dim)
        keys = self.W_key(x)        # (b, T, num_kv_groups * head_dim)
        values = self.W_value(x)    # (b, T, num_kv_groups * head_dim)
        # Reshape
        queries = queries.view(batch_size, num_tokens, self.n_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(batch_size, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)
        values = values.view(batch_size, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)
        # Expand K and V to match number of heads（先扩展，偏置才能按每头注入）
        keys = _expand_grouped_kv(keys, self.group_size, self.n_heads)
        values = _expand_grouped_kv(values, self.group_size, self.n_heads)
        # uvqk 偏置：每头每维，注入未旋转空间的 q/k/v（u 注入下方注意力输出）
        queries = queries + self.q_bias.to(queries.dtype).view(1, self.n_heads, 1, self.head_dim)
        keys = keys + self.k_bias.to(keys.dtype).view(1, self.n_heads, 1, self.head_dim)
        values = values + self.v_bias.to(values.dtype).view(1, self.n_heads, 1, self.head_dim)
        # CAT 位置编码（替代 RoPE，作用于 q/k）
        queries = self.cat(queries)
        keys = self.cat(keys)
        # Attention
        attn_scores = queries @ keys.transpose(2, 3) * self.scaling
        attn_scores = attn_scores.masked_fill(mask, -torch.inf)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        # context vector（u_bias 注入每头注意力输出）
        context_vec = attn_weights @ values + self.u_bias.to(values.dtype).view(1, self.n_heads, 1, self.head_dim)
        context_vec = context_vec.transpose(1, 2).reshape(batch_size, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)

        return context_vec


class TransformerBlockAya(nn.Module):
    """
    Tiny Aya 解码层：pre-norm RMSNorm +（CAT + uvqk）GQA + SwiGLU FFN。
    """

    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.attn = GroupedQueryAttentionAya(
            d_model=cfg.embed_dim,
            n_heads=cfg.n_heads,
            num_kv_groups=cfg.n_kv_groups,
            head_dim=cfg.head_dim,
            theta_base=cfg.rope_base,
            context_length=cfg.context_length,
            dtype=cfg.dtype,
        )
        self.ff = FeedForwardSiLU(cfg)
        self.norm1 = RMSNorm(cfg.embed_dim, eps=1e-5)
        self.norm2 = RMSNorm(cfg.embed_dim, eps=1e-5)

    def forward(self, x, mask):
        # Shortcut connection for attention block
        shortcut = x
        x = self.norm1(x)
        x = self.attn(x, mask)  # Shape: [batch_size, num_tokens, emb_size]
        x = x + shortcut
        # Shortcut connection for feed-forward block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = x + shortcut

        return x


class Model(nn.Module):
    """
    Tiny Aya 教学模型：GQA + CAT 位置编码 + uvqk 偏置 + SwiGLU +
    RMSNorm + weight tying（out_head 与 tok_embed 共享权重）。
    """

    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        # main model parameters（Command 家族风格：嵌入不做 sqrt 缩放）
        self.tok_embed = nn.Embedding(cfg.vocab_size, cfg.embed_dim, dtype=cfg.dtype)
        self.blocks = nn.ModuleList([
            TransformerBlockAya(cfg) for _ in range(cfg.n_layers)
        ])
        self.final_norm = RMSNorm(cfg.embed_dim, eps=1e-5)
        self.out_head = nn.Linear(cfg.embed_dim, cfg.vocab_size, bias=False, dtype=cfg.dtype)
        # 权重共享：weight tying
        self.out_head.weight = self.tok_embed.weight
        # causal mask
        mask = torch.triu(
            torch.ones((cfg.context_length, cfg.context_length), dtype=torch.bool),
            diagonal=1,
        )
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x):
        # Forward pass
        batch_size, seq_len = x.shape
        x = self.tok_embed(x)
        mask = self.mask[:seq_len, :seq_len]
        for block in self.blocks:
            x = block(x, mask)
        x = self.final_norm(x)
        logits = self.out_head(x)

        return logits




# 测试代码 main 函数
def main():
    from utils.log_util import logger
    from utils.args_tools import DotDict
    from config.aya.model_cfgs import AYA_TINY_CONFIG

    # ------------------------------
    # parms
    # ------------------------------
    cfg = DotDict(AYA_TINY_CONFIG)
    # ------------------------------
    # random seed
    # ------------------------------
    torch.manual_seed(123)
    # ------------------------------
    # model
    # ------------------------------
    model = Model(cfg)
    logger.info(model)
    # ------------------------------
    # model test
    # ------------------------------
    model_output = model(torch.tensor([1, 2, 3]).unsqueeze(0))
    logger.info(f"model_output.shape: {model_output.shape}")
    # ------------------------------
    # model memory
    # ------------------------------
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total number of parameters: {total_params:,}")
    total_params_normalized = total_params - model.tok_embed.weight.numel()
    logger.info(f"Total number of unique parameters: {total_params_normalized:,}")
    logger.info(f"weight tying: {model.out_head.weight is model.tok_embed.weight}")


if __name__ == "__main__":
    main()
