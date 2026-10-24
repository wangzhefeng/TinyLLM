# -*- coding: utf-8 -*-

# ***************************************************
# * File        : qwen3_5.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2026-10-24
# * Version     : 1.0.102401
# * Description : Qwen3.5（Qwen, 2026）文本栈教学实现（模型 + 配置见
# *               config/qwen3/model_cfgs.py 的 "Qwen3.5-4B"；不做权重下载）。
# *               核心差异（对照 Qwen3）：3:1 混合层——
# *               - linear_attention 层（3/4）：gated delta net 线性注意力
# *                 （qkv 联合投影 -> 因果 depthwise conv1d(k=4) -> L2 归一
# *                 gated delta rule 递推 -> SiLU 门控 RMSNorm），O(1) 推断状态
# *               - full_attention 层（每 4 层 1 层）：GQA + 逐 head 零中心
# *                 (1+w) QK-norm + sigmoid 输出门控（q_proj 双份输出）+
# *                 partial RoPE（仅前 25% head 维度旋转）
# *               - 全体块使用零中心 RMSNorm（(1+w)，权重零初始化）
# *               官方 config.json 已核实（Qwen/Qwen3.5-4B）；实现对照
# *               transformers modular_qwen3_5.py 与 qwen3_next 参考实现
# *               （gated delta rule 采用与其串行参考版数学等价的递推）。
# *               范围限定：多模态视觉塔、MTP 头、mrope 多模态位置与
# *               kernel 化 chunk 扫描不在本教学实现内（文本输入下 mrope
# *               三段位置相同，退化为普通 partial RoPE）。
# * Link        : https://huggingface.co/Qwen/Qwen3.5-4B
# *               https://github.com/huggingface/transformers/tree/main/src/transformers/models/qwen3_5
# * Requirement : torch
# ***************************************************

__all__ = [
    "l2norm",
    "gated_delta_rule",
    "RMSNormGated",
    "GatedDeltaNet_Qwen3_5",
    "GroupedQueryAttention_Qwen3_5",
    "TransformerBlockQwen3_5",
    "Model",
]

# python libraries
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.attention import _expand_grouped_kv
from layers.feed_forward import FeedForwardSiLU
from layers.kv_cache import KVCache
from layers.normalization.rms_norm import RMSNorm_Gemma3
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


def l2norm(x, eps = 1e-6):
    """FLA 对齐的 L2 归一化（q/k 进 delta rule 前使用）。"""
    return x * torch.rsqrt((x * x).sum(dim = -1, keepdim = True) + eps)


def gated_delta_rule(query, key, value, g, beta, initial_state = None):
    """
    gated delta rule 串行递推（transformers ``torch_recurrent_gated_delta_rule``
    的教学重写；官方训练/推断使用数学等价的分块并行 kernel）。

    每 token 每头三步：
      1. 状态衰减：``S <- S * exp(g_t)``
      2. delta 更新：``S <- S + k_t ⊗ β_t (v_t − k_tᵀS)``
         （只写入旧状态未能预测的修正量，即 delta rule）
      3. 读出：``o_t = q_tᵀ S``
    q/k 先做 L2 归一化，q 额外乘 ``head_k_dim**-0.5``。

    Args:
        query/key: (B, T, H, K)；value: (B, T, H, V)；g/beta: (B, T, H)
        initial_state: (B, H, K, V) 或 None

    Returns:
        (out, final_state)：out (B, T, H, V)；final_state (B, H, K, V)，fp32
    """
    dtype = query.dtype
    batch_size, seq_len, _, head_k_dim = key.shape
    num_v_heads, head_v_dim = value.shape[-2:]
    # 统一到 fp32 与 (B, H, T, D) 布局（对齐官方参考实现的数值精度）
    query, key, value = (t.transpose(1, 2).float() for t in (query, key, value))
    decay = g.transpose(1, 2).float()
    beta = beta.transpose(1, 2).float()
    query = l2norm(query) * (head_k_dim ** -0.5)
    key = l2norm(key)
    if initial_state is None:
        state = query.new_zeros(batch_size, num_v_heads, head_k_dim, head_v_dim)
    else:
        state = initial_state.float()
    outputs = []
    for t in range(seq_len):
        q_t, k_t, v_t = query[:, :, t], key[:, :, t], value[:, :, t]
        # 1. 状态衰减
        state = state * decay[:, :, t].exp()[..., None, None]
        # 2. delta 更新
        kv_mem = torch.einsum("bhkv,bhk->bhv", state, k_t)
        delta = (v_t - kv_mem) * beta[:, :, t].unsqueeze(-1)
        state = state + torch.einsum("bhk,bhv->bhkv", k_t, delta)
        # 3. 读出
        outputs.append(torch.einsum("bhkv,bhk->bhv", state, q_t))
    core_attn_out = torch.stack(outputs, dim = 2).to(dtype)  # (B, H, T, V)

    return core_attn_out.transpose(1, 2), state


class RMSNormGated(nn.Module):
    """
    gated delta net 输出归一化：``weight * RMSNorm(x) * SiLU(z)``。

    weight 服从官方 ones 初始化；z 为与输出同形状的门控向量。
    """

    def __init__(self, dim, eps = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x, z):
        input_dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(dim = -1, keepdim = True) + self.eps)
        x = (self.weight * x) * F.silu(z.float())

        return x.to(input_dtype)


class GatedDeltaNet_Qwen3_5(nn.Module):
    """
    Qwen3.5 线性注意力层（对照 transformers ``Qwen3_5GatedDeltaNet``）。

    结构：qkv 联合投影 + 独立 z/b/a 投影 -> 因果 depthwise conv1d（核 4，
    左补 k-1）-> q/k L2 归一 gated delta rule 递推 -> SiLU 门控 RMSNorm
    -> 输出投影。推断状态为 (conv_state, recurrent_state)，与 full 层的
    (k, v) 一起由外部 ``KVCache`` 透明存储。

    与 Qwen3-Next 的差异：qkvz/ba 两个联合投影拆分为 qkv/z/b/a 四个独立投影。
    """

    def __init__(self, d_model, num_k_heads, head_k_dim, num_v_heads, head_v_dim,
                 conv_kernel = 4, dtype = None):
        super().__init__()

        self.num_k_heads = num_k_heads
        self.head_k_dim = head_k_dim
        self.num_v_heads = num_v_heads
        self.head_v_dim = head_v_dim
        self.key_dim = num_k_heads * head_k_dim
        self.value_dim = num_v_heads * head_v_dim
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv_kernel = conv_kernel
        # 投影（无 bias）
        self.in_proj_qkv = nn.Linear(d_model, self.conv_dim, bias = False, dtype = dtype)
        self.in_proj_z = nn.Linear(d_model, self.value_dim, bias = False, dtype = dtype)
        self.in_proj_b = nn.Linear(d_model, self.num_v_heads, bias = False, dtype = dtype)
        self.in_proj_a = nn.Linear(d_model, self.num_v_heads, bias = False, dtype = dtype)
        # 因果 depthwise 卷积（padding=0，历史由 forward 手工左补）
        self.conv1d = nn.Conv1d(
            self.conv_dim, self.conv_dim,
            kernel_size = conv_kernel,
            groups = self.conv_dim,
            bias = False,
            padding = 0,
            dtype = dtype,
        )
        # 门控/衰减参数（官方初始化：dt_bias=1，A=exp(A_log)~U(0.01,16)）
        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))
        self.A_log = nn.Parameter(
            torch.log(torch.empty(self.num_v_heads).uniform_(0.01, 16.0))
        )
        self.norm = RMSNormGated(self.head_v_dim)
        self.out_proj = nn.Linear(self.value_dim, d_model, bias = False, dtype = dtype)

    def forward(self, x, cache = None):
        batch_size, seq_len, _ = x.shape
        if cache is not None:
            conv_state, recurrent_state = cache
        else:
            conv_state, recurrent_state = None, None
        # 投影
        mixed_qkv = self.in_proj_qkv(x).transpose(1, 2)      # (B, conv_dim, T)
        z = self.in_proj_z(x)
        b = self.in_proj_b(x)
        a = self.in_proj_a(x)
        # 因果 depthwise conv：左侧补 k-1 个历史输入（cache 提供真实历史，
        # 等价于官方对称 padding 后截取的因果卷积）
        if conv_state is not None:
            mixed_qkv = torch.cat([conv_state.to(mixed_qkv.dtype), mixed_qkv], dim = -1)
        else:
            mixed_qkv = F.pad(mixed_qkv, (self.conv_kernel - 1, 0))
        new_conv_state = mixed_qkv[:, :, -(self.conv_kernel - 1):]
        mixed_qkv = F.silu(self.conv1d(mixed_qkv)).transpose(1, 2)
        # 拆分 q/k/v 并重排 heads
        query, key, value = torch.split(
            mixed_qkv,
            [self.key_dim, self.key_dim, self.value_dim],
            dim = -1,
        )
        query = query.view(batch_size, seq_len, self.num_k_heads, self.head_k_dim)
        key = key.view(batch_size, seq_len, self.num_k_heads, self.head_k_dim)
        value = value.view(batch_size, seq_len, self.num_v_heads, self.head_v_dim)
        # 门控与衰减率
        beta = torch.sigmoid(b)
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
        # k 头数少于 v 头数时，q/k 沿头维复制对齐
        if self.num_v_heads // self.num_k_heads > 1:
            repeat = self.num_v_heads // self.num_k_heads
            query = query.repeat_interleave(repeat, dim = 2)
            key = key.repeat_interleave(repeat, dim = 2)
        # gated delta rule 递推
        core_attn_out, last_state = gated_delta_rule(
            query, key, value, g, beta,
            initial_state = recurrent_state,
        )
        # SiLU 门控 RMSNorm（逐 v-head）+ 输出投影
        core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
        z = z.view(batch_size, seq_len, self.num_v_heads, self.head_v_dim)
        z = z.reshape(-1, self.head_v_dim)
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.view(batch_size, seq_len, self.value_dim)
        output = self.out_proj(core_attn_out)

        return output, (new_conv_state, last_state)


def compute_partial_rope(x, cos, sin, rotary_dim, offset = 0):
    """
    partial RoPE：只旋转 head 维度前 ``rotary_dim`` 个通道，其余直通。

    Qwen3.5 的 rotary_dim = head_dim * partial_rotary_factor（4B 为 25%，
    即 256 维中前 64 维参与旋转）；cos/sin 按 rotary_dim 预计算。
    """
    if rotary_dim >= x.shape[-1]:
        return compute_rope(x, cos, sin, offset = offset)
    x_rot = compute_rope(x[..., :rotary_dim], cos, sin, offset = offset)

    return torch.cat([x_rot, x[..., rotary_dim:]], dim = -1)


class GroupedQueryAttention_Qwen3_5(nn.Module):
    """
    Qwen3.5 full-attention 层（对照 transformers ``Qwen3_5Attention``）。

    与 Qwen3 full attention 的差异：
    - 输出门控：q_proj 输出 2 倍宽度，按 head 拆为 query 与 gate，
      attn 输出乘 ``sigmoid(gate)`` 后再过 o_proj
    - 逐 head 零中心 (1+w) QK-norm（权重零初始化，Qwen3 为 ones 初始化）
    - partial RoPE：仅前 rotary_dim 个通道旋转
    """

    def __init__(self, d_model, n_heads, num_kv_groups, head_dim = None,
                 attn_output_gate = True, rotary_dim = None, dtype = None):
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
        self.attn_output_gate = attn_output_gate
        self.rotary_dim = head_dim if rotary_dim is None else rotary_dim
        # query（含门控时 2 倍宽度）/ key / value / 输出投影
        query_width = self.d_out * 2 if attn_output_gate else self.d_out
        self.W_query = nn.Linear(d_model, query_width, bias = False, dtype = dtype)
        self.W_key = nn.Linear(d_model, num_kv_groups * head_dim, bias = False, dtype = dtype)
        self.W_value = nn.Linear(d_model, num_kv_groups * head_dim, bias = False, dtype = dtype)
        self.out_proj = nn.Linear(self.d_out, d_model, bias = False, dtype = dtype)
        # 逐 head QK-norm（零中心 (1+w)，权重零初始化）
        self.q_norm = RMSNorm_Gemma3(head_dim, eps = 1e-6)
        self.k_norm = RMSNorm_Gemma3(head_dim, eps = 1e-6)

    def forward(self, x, mask, cos, sin, start_pos = 0, cache = None):
        batch_size, num_tokens, _ = x.shape
        # 投影；门控开启时 query 与 gate 按 head 交错拆分
        q_out = self.W_query(x)
        if self.attn_output_gate:
            q_out = q_out.view(batch_size, num_tokens, self.n_heads, 2 * self.head_dim)
            query, gate = torch.chunk(q_out, 2, dim = -1)
            gate = gate.reshape(batch_size, num_tokens, -1)
        else:
            query = q_out.view(batch_size, num_tokens, self.n_heads, self.head_dim)
            gate = None
        key = self.W_key(x).view(batch_size, num_tokens, self.num_kv_groups, self.head_dim)
        value = self.W_value(x).view(batch_size, num_tokens, self.num_kv_groups, self.head_dim)
        # 逐 head QK-norm 后重排为 (B, H, T, D)
        query = self.q_norm(query).transpose(1, 2)
        key = self.k_norm(key).transpose(1, 2)
        value = value.transpose(1, 2)
        # partial RoPE（cache 增量时按绝对位置偏移）
        query = compute_partial_rope(query, cos, sin, self.rotary_dim, offset = start_pos)
        key = compute_partial_rope(key, cos, sin, self.rotary_dim, offset = start_pos)
        # KV cache
        if cache is not None:
            prev_k, prev_v = cache
            keys = torch.cat([prev_k, key], dim = 2)
            values = torch.cat([prev_v, value], dim = 2)
            next_cache = (keys, values)
        else:
            keys, values = key, value
            next_cache = (keys, values)
        # Expand K and V to match number of heads
        keys = _expand_grouped_kv(keys, self.group_size, self.n_heads)
        values = _expand_grouped_kv(values, self.group_size, self.n_heads)
        # Attention
        attn_scores = query @ keys.transpose(2, 3)
        attn_scores = attn_scores.masked_fill(mask, -torch.inf)
        attn_weights = torch.softmax(attn_scores / self.head_dim ** 0.5, dim = -1)
        # context vector + sigmoid 输出门控
        context_vec = (attn_weights @ values).transpose(1, 2).reshape(batch_size, num_tokens, self.d_out)
        if gate is not None:
            context_vec = context_vec * torch.sigmoid(gate)
        context_vec = self.out_proj(context_vec)

        return context_vec, next_cache


class TransformerBlockQwen3_5(nn.Module):
    """
    Qwen3.5 解码块（标准 pre-norm）：norm -> token mixer -> 残差；
    norm -> SwiGLU FFN -> 残差。token mixer 按层类型二选一：
    linear_attention（gated delta net，无需 mask）或 full_attention。
    """

    def __init__(self, cfg, layer_type):
        super().__init__()

        self.layer_type = layer_type
        self._compute_dtype = cfg.dtype

        if layer_type == "linear_attention":
            self.attn = GatedDeltaNet_Qwen3_5(
                d_model = cfg.embed_dim,
                num_k_heads = _cfg_get(cfg, "linear_num_key_heads", cfg.n_heads),
                head_k_dim = _cfg_get(cfg, "linear_key_head_dim", cfg.head_dim),
                num_v_heads = _cfg_get(cfg, "linear_num_value_heads", cfg.n_heads),
                head_v_dim = _cfg_get(cfg, "linear_value_head_dim", cfg.head_dim),
                conv_kernel = _cfg_get(cfg, "linear_conv_kernel_dim", 4),
                dtype = cfg.dtype,
            )
        else:
            head_dim = cfg.head_dim if cfg.head_dim is not None else cfg.embed_dim // cfg.n_heads
            self.attn = GroupedQueryAttention_Qwen3_5(
                d_model = cfg.embed_dim,
                n_heads = cfg.n_heads,
                num_kv_groups = cfg.n_kv_groups,
                head_dim = cfg.head_dim,
                attn_output_gate = _cfg_get(cfg, "attn_output_gate", False),
                rotary_dim = int(head_dim * _cfg_get(cfg, "partial_rotary_factor", 1.0)),
                dtype = cfg.dtype,
            )
        self.ff = FeedForwardSiLU(cfg)
        # 零中心 (1+w) RMSNorm，权重零初始化（Qwen3.5/3-Next 风格）
        self.input_layernorm = RMSNorm_Gemma3(cfg.embed_dim, eps = 1e-6)
        self.post_attention_layernorm = RMSNorm_Gemma3(cfg.embed_dim, eps = 1e-6)

    def forward(self, x, mask, cos, sin, start_pos = 0, cache = None):
        # Shortcut connection for attention block
        residual = x
        x = self.input_layernorm(x)
        if self.layer_type == "linear_attention":
            # 递推天然因果，无需 mask
            x, next_cache = self.attn(x.to(self._compute_dtype), cache = cache)
        else:
            x, next_cache = self.attn(
                x.to(self._compute_dtype), mask, cos, sin,
                start_pos = start_pos,
                cache = cache,
            )
        x = residual + x

        # Shortcut connection for feed-forward block
        residual = x
        x = self.post_attention_layernorm(x)
        x = self.ff(x.to(self._compute_dtype))
        x = residual + x

        return x, next_cache


class Model(nn.Module):
    """
    Qwen3.5 解码器（文本栈教学实现）。

    ``full_attention_interval=4`` 时第 4/8/12... 层为 full attention，
    其余层为 gated delta net 线性注意力（对照 Qwen3：全部为 full attention）。
    KVCache 对两类层透明：full 层存 (k, v)，线性层存 (conv_state, recurrent_state)。
    """

    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        # Embedding
        self.tok_embed = nn.Embedding(cfg.vocab_size, cfg.embed_dim, dtype = cfg.dtype)
        # transformer block
        interval = _cfg_get(cfg, "full_attention_interval", 0)
        self.layer_types = [
            "full_attention" if interval > 0 and (i + 1) % interval == 0 else "linear_attention"
            for i in range(cfg.n_layers)
        ]
        self.trf_blocks = nn.ModuleList(
            [TransformerBlockQwen3_5(cfg, layer_type) for layer_type in self.layer_types]
        )
        # RMSNorm
        self.final_norm = RMSNorm_Gemma3(cfg.embed_dim, eps = 1e-6)
        # output head linear
        self.out_head = nn.Linear(cfg.embed_dim, cfg.vocab_size, bias = False, dtype = cfg.dtype)
        # weight tying（Qwen3.5-4B 官方开启）
        if _cfg_get(cfg, "tie_word_embeddings", False):
            self.out_head.weight = self.tok_embed.weight
        # head dim 与 partial RoPE 的旋转维数
        if cfg.head_dim is None:
            head_dim = cfg.embed_dim // cfg.n_heads
        else:
            head_dim = cfg.head_dim
        self.rotary_dim = int(head_dim * _cfg_get(cfg, "partial_rotary_factor", 1.0))
        assert self.rotary_dim > 0 and self.rotary_dim % 2 == 0, \
            "rotary_dim (head_dim * partial_rotary_factor) must be a positive even number"
        # RoPE params（按 rotary_dim 预计算）
        cos, sin = precompute_rope_params(
            head_dim = self.rotary_dim,
            theta_base = cfg.rope_base,
            context_length = cfg.context_length,
        )
        self.register_buffer("cos", cos, persistent = False)
        self.register_buffer("sin", sin, persistent = False)
        self.current_pos = 0

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
        # full-attention 层的因果 mask（线性注意力层递推天然因果，不消费 mask）
        key_len = pos_end if cache is not None else num_tokens
        row = torch.arange(pos_start, pos_end, device = x.device).unsqueeze(1)
        col = torch.arange(key_len, device = x.device).unsqueeze(0)
        mask = col > row
        if attention_mask is not None:
            if attention_mask.ndim == 2:
                attention_mask = attention_mask[:, None, None, :]
            mask = mask[None, None, :, :] | attention_mask.bool()
        else:
            mask = mask[None, None, :, :]
        # forward
        x = self.tok_embed(x)
        for i, block in enumerate(self.trf_blocks):
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
    from types import SimpleNamespace

    import torch

    from utils.device import device_setting
    from utils.model_memory import model_memory_size
    from utils.log_util import logger

    # random seed
    torch.manual_seed(123)

    # device
    device = device_setting(verbose = True)

    # tiny 配置（结构对照 Qwen3.5-4B 等比缩小；正式配置见 config/qwen3/model_cfgs.py）
    QWEN3_5_TINY = SimpleNamespace(
        vocab_size = 64, context_length = 32, embed_dim = 32, d_ff = 64,
        n_heads = 4, n_layers = 4, n_kv_groups = 2, head_dim = 16,
        qk_norm = True, rope_base = 10_000_000.0,
        num_experts = 0, dtype = torch.float32,
        full_attention_interval = 4, attn_output_gate = True,
        partial_rotary_factor = 0.5,
        linear_num_key_heads = 2, linear_key_head_dim = 8,
        linear_num_value_heads = 4, linear_value_head_dim = 8,
        linear_conv_kernel_dim = 4,
        tie_word_embeddings = True,
    )

    # model
    model = Model(cfg = QWEN3_5_TINY)
    model.to(device)
    logger.info(f"model: \n{model}")
    logger.info(f"layer_types: {model.layer_types}")

    # model forward
    input_tensor = torch.tensor([1, 2, 3]).unsqueeze(0)
    input_tensor = input_tensor.to(device)
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
