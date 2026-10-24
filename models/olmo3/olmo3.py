# -*- coding: utf-8 -*-

# ***************************************************
# * File        : olmo3.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2026-10-24
# * Version     : 1.0.102401
# * Description : Olmo 3 (allenai, 2025) 教学实现（模型 + 配置见
# *               config/olmo3/model_cfgs.py；不做权重下载）。
# *               与 OLMo 2 相同的骨干，混合注意力为其核心差异：
# *               - post-norm block：attn 输出先 norm 再残差相加，
# *                 FFN 输出先 norm 再残差相加（非 Llama/Gemma 的 pre-norm）
# *               - QK-norm：作用在整个 q/k 投影输出向量（flat，非逐头），
# *                 且为非参数化 RMSNorm（无可学习权重）
# *               - 3:1 混合层：每 4 层 1 层 full attention，
# *                 其余 3 层 sliding-window(4096) attention
# *               - RoPE theta=500000，RMSNorm eps=1e-5，全程无 bias，SwiGLU FFN
# *               架构已对照 transformers 官方源码核实（models/olmo3/
# *               {configuration,modular}_olmo3.py、models/olmo2/modeling_olmo2.py）。
# *               尺寸超参未经 checkpoint 级 config.json 核实（gated repo），
# *               见 config/olmo3/model_cfgs.py 头注释。
# * Link        : https://huggingface.co/allenai/OLMo-3-7B-Instruct
# *               https://arxiv.org/abs/2501.00656 (OLMo 2)
# * Requirement : torch
# ***************************************************

__all__ = [
    "NonParametricRMSNorm",
    "GroupedQueryAttention_Olmo3",
    "TransformerBlockOlmo3",
    "Model",
]

# python libraries
from pathlib import Path

import torch
import torch.nn as nn

from layers.attention import _expand_grouped_kv
from layers.feed_forward import FeedForwardSiLU
from layers.kv_cache import KVCache
from layers.normalization.rms_norm import RMSNorm
from layers.position_encoding.RoPE import compute_rope, precompute_rope_params

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


def _cfg_get(cfg, name, default = None):
    """从 ModelConfig / SimpleNamespace / dict 风格配置中安全取键，缺省返回 default。"""
    try:
        value = cfg[name] if isinstance(cfg, dict) else getattr(cfg, name)
    except (AttributeError, KeyError, TypeError):
        return default
    return default if value is None else value


class NonParametricRMSNorm(nn.Module):
    """
    非参数化 RMSNorm：``x / RMS(x)``，无可学习权重。

    Olmo 2/3 的 QK-norm 使用此形式（HF 实现为权重恒 1 初始化的 RMSNorm，
    初始时二者等价）；作用域为整个投影输出向量，而非单个 head。
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x):
        input_dtype = x.dtype
        x = x.float()
        variance = x.pow(2).mean(dim = -1, keepdim = True)
        return (x * torch.rsqrt(variance + self.eps)).to(input_dtype)


class GroupedQueryAttention_Olmo3(nn.Module):
    """
    Olmo 3 GQA：投影 -> flat 级非参数 QK-norm -> 重排 heads -> RoPE -> 注意力。

    与 ``GroupedQueryAttention_Qwen3`` 的差异：QK-norm 为非参数化且作用在
    整个投影输出上（Qwen3 为逐 head、带权重）；滑动窗口由外部 mask 实现
    （见 ``Model.forward``），层本身对 full/sliding 通用。
    """

    def __init__(self, d_model, n_heads, num_kv_groups, head_dim = None, dtype = None):
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
        # query, key, value weights
        self.W_query = nn.Linear(d_model, self.d_out, bias = False, dtype = dtype)
        self.W_key = nn.Linear(d_model, num_kv_groups * head_dim, bias = False, dtype = dtype)
        self.W_value = nn.Linear(d_model, num_kv_groups * head_dim, bias = False, dtype = dtype)
        # out projection
        self.out_proj = nn.Linear(self.d_out, d_model, bias = False, dtype = dtype)
        # QK-norm：flat 级（覆盖全部 heads）、非参数化
        self.q_norm = NonParametricRMSNorm(self.d_out, eps = 1e-5)
        self.k_norm = NonParametricRMSNorm(num_kv_groups * head_dim, eps = 1e-5)

    def forward(self, x, mask, cos, sin, start_pos = 0, cache = None):
        batch_size, num_tokens, _ = x.shape
        # 投影 + flat 级非参数 QK-norm（先 norm 再重排 heads）
        queries = self.q_norm(self.W_query(x))
        keys_new = self.k_norm(self.W_key(x))
        values_new = self.W_value(x)
        # 重排为 (batch_size, n_heads/groups, num_tokens, head_dim)
        queries = queries.view(batch_size, num_tokens, self.n_heads, self.head_dim).transpose(1, 2)
        keys_new = keys_new.view(batch_size, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)
        values_new = values_new.view(batch_size, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)
        # RoPE（cache 增量时按绝对位置偏移）
        queries = compute_rope(queries, cos, sin, offset = start_pos)
        keys_new = compute_rope(keys_new, cos, sin, offset = start_pos)
        # KV cache
        if cache is not None:
            prev_k, prev_v = cache
            keys = torch.cat([prev_k, keys_new], dim = 2)
            values = torch.cat([prev_v, values_new], dim = 2)
            next_cache = (keys, values)
        else:
            keys, values = keys_new, values_new
            next_cache = (keys, values)
        # Expand K and V to match number of heads
        keys = _expand_grouped_kv(keys, self.group_size, self.n_heads)
        values = _expand_grouped_kv(values, self.group_size, self.n_heads)
        # Attention（标准 1/sqrt(d) 缩放；QK-norm 之后仍保留，同官方实现）
        attn_scores = queries @ keys.transpose(2, 3)
        attn_scores = attn_scores.masked_fill(mask, -torch.inf)
        attn_weights = torch.softmax(attn_scores / self.head_dim ** 0.5, dim = -1)
        # context vector
        context_vec = (attn_weights @ values).transpose(1, 2).reshape(batch_size, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)

        return context_vec, next_cache


class TransformerBlockOlmo3(nn.Module):
    """
    Olmo 3 post-norm block：分支输出先 norm，再与残差相加。

    对照（均为 pre-norm）：Llama/Gemma/Qwen 为 norm -> 分支 -> 相加；
    Olmo 2/3 为 分支 -> norm -> 相加（block 尾部无 norm，模型末尾有 final norm）。
    """

    def __init__(self, cfg):
        super().__init__()

        self._compute_dtype = cfg.dtype

        self.attn = GroupedQueryAttention_Olmo3(
            d_model = cfg.embed_dim,
            n_heads = cfg.n_heads,
            num_kv_groups = cfg.n_kv_groups,
            head_dim = cfg.head_dim,
            dtype = cfg.dtype,
        )
        self.ff = FeedForwardSiLU(cfg)
        self.post_attention_norm = RMSNorm(cfg.embed_dim, eps = 1e-5)
        self.post_feedforward_norm = RMSNorm(cfg.embed_dim, eps = 1e-5)

    def forward(self, x, mask, cos, sin, start_pos = 0, cache = None):
        residual = x
        attn_out, next_cache = self.attn(
            x.to(self._compute_dtype), mask, cos, sin,
            start_pos = start_pos,
            cache = cache,
        )
        x = residual + self.post_attention_norm(attn_out)

        residual = x
        ff_out = self.ff(x.to(self._compute_dtype))
        x = residual + self.post_feedforward_norm(ff_out)

        return x, next_cache


class Model(nn.Module):
    """
    Olmo 3 解码器（教学实现）。

    3:1 混合注意力：``full_attention_interval=4`` 时第 4/8/12... 层为
    full attention，其余层为 sliding-window attention；两类层 RoPE 相同
    （theta=500000），仅 mask 不同（官方对滑窗层不施加额外 RoPE 缩放）。
    """

    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        # Embedding
        self.tok_embed = nn.Embedding(cfg.vocab_size, cfg.embed_dim, dtype = cfg.dtype)
        # transformer block
        interval = _cfg_get(cfg, "full_attention_interval", 0)
        self.layer_types = [
            "full_attention" if interval > 0 and (i + 1) % interval == 0 else "sliding_attention"
            for i in range(cfg.n_layers)
        ]
        self.trf_blocks = nn.ModuleList(
            [TransformerBlockOlmo3(cfg) for _ in range(cfg.n_layers)]
        )
        # RMSNorm
        self.final_norm = RMSNorm(cfg.embed_dim, eps = 1e-5)
        # output head linear
        self.out_head = nn.Linear(cfg.embed_dim, cfg.vocab_size, bias = False, dtype = cfg.dtype)
        # weight tying
        if _cfg_get(cfg, "tie_word_embeddings", False):
            self.out_head.weight = self.tok_embed.weight
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
        self.register_buffer("cos", cos, persistent = False)
        self.register_buffer("sin", sin, persistent = False)
        self.current_pos = 0
        # sliding window（None 时 sliding_attention 层退化为纯因果）
        self.sliding_window = _cfg_get(cfg, "sliding_window", None)

    def forward(self, x, cache = None, attention_mask = None, position_ids = None):
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
        # 逐层 mask：full 层为因果 mask，sliding 层额外屏蔽窗口之外的历史
        # （row/col 均为绝对位置，天然兼容 KV cache 增量）
        key_len = pos_end if cache is not None else num_tokens
        row = torch.arange(pos_start, pos_end, device = x.device).unsqueeze(1)
        col = torch.arange(key_len, device = x.device).unsqueeze(0)
        causal = col > row                                # True = 屏蔽（未来）
        if attention_mask is not None:
            if attention_mask.ndim == 2:
                attention_mask = attention_mask[:, None, None, :]
            causal = causal[None, None, :, :] | attention_mask.bool()
        else:
            causal = causal[None, None, :, :]
        full_mask = causal
        if self.sliding_window is not None:
            swa_mask = causal | (row - col >= self.sliding_window)[None, None, :, :]
        else:
            swa_mask = full_mask
        # forward
        x = self.tok_embed(x)
        for i, block in enumerate(self.trf_blocks):
            mask = swa_mask if self.layer_types[i] == "sliding_attention" else full_mask
            block_cache = cache.get(i) if cache is not None else None
            x, new_block_cache = block(
                x, mask, self.cos, self.sin,
                start_pos = pos_start,
                cache = block_cache,
            )
            if cache is not None:
                cache.update(i, new_block_cache)
        if cache is not None:
            cache.advance(num_tokens)
        self.current_pos = pos_end

        return self.out_head(self.final_norm(x).to(self.out_head.weight.dtype))

    def reset_kv_cache(self):
        self.current_pos = 0

    def new_cache(self):
        return KVCache(self.cfg.n_layers)





# 测试代码 main 函数
def main():
    import torch

    from utils.device import device_setting
    from utils.model_memory import model_memory_size
    from config.olmo3.model_cfgs import get_cfgs
    from utils.log_util import logger

    # random seed
    torch.manual_seed(123)

    # device
    device = device_setting(verbose = True)

    # model
    OLMO3_TINY = get_cfgs(CHOOSE_MODEL = "TINY")
    model = Model(cfg = OLMO3_TINY)
    model.to(device)
    logger.info(f"model: \n{model}")

    # model forward
    input_tensor = torch.tensor([1, 2, 3]).unsqueeze(0)
    input_tensor = input_tensor.to(device)
    logger.info(f"input_tensor shape: {input_tensor.shape}")

    output_tensor = model(input_tensor)
    logger.info(f"output_tensor shape: {output_tensor.shape}")

    # KV cache 增量前向
    cache = model.new_cache()
    model(input_tensor, cache = cache)
    incremental = model(torch.tensor([[4]]).to(device), cache = cache)
    logger.info(f"incremental output shape: {incremental.shape}")

    model_memory_size(model, input_dtype = torch.float32, verbose = True)
    model_memory_size(model, input_dtype = torch.bfloat16, verbose = True)

if __name__ == "__main__":
    main()
