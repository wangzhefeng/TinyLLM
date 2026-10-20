# -*- coding: utf-8 -*-

# ***************************************************
# * File        : attention.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-07-27
# * Version     : 1.1.072816
# * Description : Qwen3Attention (继承自 nn.Module)
# *               ├── qkv_proj: QKVParallelLinear (用于合并计算 Q, K, V)
# *               │   └── (继承自) ColumnParallelLinear (列并行层)
# *               ├── o_proj: RowParallelLinear (用于 Attention 输出)
# *               ├── rotary_emb: RotaryEmbedding (旋转位置编码)
# *               ├── attn: Attention (底层 Attention 计算核心)
# *               │   └── (使用) Context (用于获取 prefill/decode 状态及相关参数)
# *               ├── q_norm / k_norm: RMSNorm (Qwen3 对 Q/K 逐头归一化)
# *               │
# *               与上游的区别：上游 Attention 使用 triton + flash-attn，
# *               本实现用 F.scaled_dot_product_attention 在 CPU 上完成
# *               prefill(varlen 因果) 和 decode(分页 KV cache 读取)。
# * Link        : https://github.com/GeeeekExplorer/nano-vllm
# * Requirement : torch >= 2.0
# ***************************************************

__all__ = [
    "Attention",
    "Qwen3Attention",
    "store_kvcache",
]

# python libraries
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)
PKG_ROOT = str(Path(__file__).resolve().parents[1])
if PKG_ROOT in sys.path:
    sys.path.remove(PKG_ROOT)
sys.path.insert(0, PKG_ROOT)
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.norm import RMSNorm
from layers.parallel_linear import QKVParallelLinear, RowParallelLinear, _tp_rank_and_size
from layers.rotary_embedding import get_rope
from utils.context import get_context

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


def store_kvcache(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
):
    """
    把本 step 的 K/V 按 slot_mapping 写入分页 KV cache(纯 PyTorch，替代上游 triton kernel)

    Args:
        key / value: shape [num_tokens, num_kv_heads, head_dim]
        k_cache / v_cache: shape [num_blocks, block_size, num_kv_heads, head_dim]
        slot_mapping: 每个 token 的目标槽位 id = block_id * block_size + offset, shape [num_tokens]
    """
    num_tokens, num_kv_heads, head_dim = key.shape
    d = num_kv_heads * head_dim
    slots = slot_mapping.long()
    k_cache.view(-1, d).index_copy_(0, slots, key.view(num_tokens, d))
    v_cache.view(-1, d).index_copy_(0, slots, value.view(num_tokens, d))


class Attention(nn.Module):
    """
    Attention 计算核心：
    - prefill: 按 cu_seqlens 切出每条序列，逐条做因果 SDPA(varlen)
    - decode: 按 block_tables 从分页 KV cache 收集历史 K/V，做一次非因果 SDPA
    """

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        scale: float,
        num_kv_heads: int,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.num_kv_groups = num_heads // num_kv_heads  # GQA 分组数
        # KV cache 由 ModelRunner 分配后挂载到这里(挂载前为空 tensor，用于 warmup)
        self.k_cache = self.v_cache = torch.tensor([])

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        """
        GQA: 将 KV 头重复扩展到与 Q 头数一致。x: [..., num_kv_heads, seq_len, head_dim]
        """
        if self.num_kv_groups == 1:
            return x
        return x.repeat_interleave(self.num_kv_groups, dim=-3)

    def _gather_kv_from_cache(self, block_table_row: torch.Tensor, seqlen_k: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        从分页 KV cache 收集某条序列前 seqlen_k 个 token 的 K/V

        Args:
            block_table_row: 该序列的 block 表(可能含 -1 padding)
            seqlen_k: 需要的上下文长度
        Returns:
            k / v: [seqlen_k, num_kv_heads, head_dim]
        """
        block_size = self.k_cache.size(1)
        num_blocks = (seqlen_k + block_size - 1) // block_size
        block_ids = block_table_row[:num_blocks].long()  # 前 num_blocks 个必然有效
        slot_ids = (block_ids.unsqueeze(-1) * block_size + torch.arange(block_size)).flatten()[:seqlen_k]
        d = self.num_kv_heads * self.head_dim
        k = self.k_cache.view(-1, d).index_select(0, slot_ids).view(seqlen_k, self.num_kv_heads, self.head_dim)
        v = self.v_cache.view(-1, d).index_select(0, slot_ids).view(seqlen_k, self.num_kv_heads, self.head_dim)
        return k, v

    def _prefill_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, context) -> torch.Tensor:
        """
        prefill: q/k/v shape [total_tokens, num_(kv_)heads, head_dim]，逐序列做因果 SDPA
        """
        cu_q = context.cu_seqlens_q.tolist()
        cu_k = context.cu_seqlens_k.tolist()
        o = torch.empty_like(q)
        for i in range(len(cu_q) - 1):
            qs, qe = cu_q[i], cu_q[i + 1]
            ks, ke = cu_k[i], cu_k[i + 1]
            len_q, len_k = qe - qs, ke - ks
            if len_k == len_q:
                # 无前缀：K/V 直接取自本 step 输入
                k_i, v_i = k[ks:ke], v[ks:ke]
            else:
                # 带缓存前缀(chunked prefill / prefix cache)：历史 K/V 从分页 cache 读取
                # (当前 chunk 的 K/V 已在 store_kvcache 中写入 cache)
                k_i, v_i = self._gather_kv_from_cache(context.block_tables[i], len_k)
            # [len, heads, head_dim] -> [1, heads, len, head_dim]
            q_i = q[qs:qe].transpose(0, 1).unsqueeze(0)
            k_i = self._repeat_kv(k_i.transpose(0, 1)).unsqueeze(0)
            v_i = self._repeat_kv(v_i.transpose(0, 1)).unsqueeze(0)
            if len_k == len_q:
                # 标准因果(prefill 整块)
                o_i = F.scaled_dot_product_attention(q_i, k_i, v_i, is_causal=True, scale=self.scale)
            else:
                # 允许看到全部历史 + 当前块内因果
                mask = torch.ones(len_q, len_k, dtype=torch.bool).tril(diagonal=len_k - len_q)
                o_i = F.scaled_dot_product_attention(q_i, k_i, v_i, attn_mask=mask, scale=self.scale)
            o[qs:qe] = o_i.squeeze(0).transpose(0, 1)
        return o

    def _decode_attention(self, q: torch.Tensor, context) -> torch.Tensor:
        """
        decode: q shape [bs, num_heads, head_dim]，从分页 KV cache 读取历史 K/V
        """
        bs = q.size(0)
        block_size = self.k_cache.size(1)
        block_tables = context.block_tables  # [bs, max_num_blocks]，padding 为 -1
        context_lens = context.context_lens  # [bs]
        max_ctx_len = int(context_lens.max().item())

        # 将 block_tables 展开为槽位 id: slot = block_id * block_size + 块内偏移
        num_blocks_row = block_tables.size(1)
        block_ids = block_tables.long().clamp(min=0)  # -1 的 padding 暂用 0 占位，随后被 mask 掉
        slot_ids = (block_ids.unsqueeze(-1) * block_size + torch.arange(block_size)).flatten()  # [bs * num_blocks_row * block_size]

        d = self.num_kv_heads * self.head_dim
        k = self.k_cache.view(-1, d).index_select(0, slot_ids)  # [bs * num_blocks_row * block_size, nkv * head_dim]
        v = self.v_cache.view(-1, d).index_select(0, slot_ids)
        total_slots = num_blocks_row * block_size
        # [bs, total_slots, nkv, head_dim] -> [bs, nkv, total_slots, head_dim]
        k = k.view(bs, total_slots, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(bs, total_slots, self.num_kv_heads, self.head_dim).transpose(1, 2)
        # 截断到本 batch 的最长上下文，减少无效计算
        k = k[:, :, :max_ctx_len]
        v = v[:, :, :max_ctx_len]
        k = self._repeat_kv(k)
        v = self._repeat_kv(v)

        # 注意力掩码：只允许看到 context_lens 内的 token(True = 可见)
        attn_mask = torch.arange(max_ctx_len).unsqueeze(0) < context_lens.unsqueeze(1)  # [bs, max_ctx]
        attn_mask = attn_mask.view(bs, 1, 1, max_ctx_len)

        # q: [bs, num_heads, head_dim] -> [bs, num_heads, 1, head_dim]
        o = F.scaled_dot_product_attention(q.unsqueeze(2), k, v, attn_mask=attn_mask, scale=self.scale)
        return o.squeeze(2)  # [bs, num_heads, head_dim]

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Args:
            q: [num_tokens, num_heads, head_dim]
            k / v: [num_tokens, num_kv_heads, head_dim]
        Returns:
            o: [num_tokens, num_heads, head_dim] (decode 时 num_tokens == batch_size)
        """
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        if context.is_prefill:
            o = self._prefill_attention(q, k, v, context)
        else:  # decode
            o = self._decode_attention(q, context)
        return o


class Qwen3Attention(nn.Module):
    """
    自注意力机制(Qwen3 风格: GQA + Q/K RMSNorm + RoPE)
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = 4096,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-6,
        qkv_bias: bool = False,
        rope_theta: float = 10000.0,
    ):
        super().__init__()
        tp_size = _tp_rank_and_size()[1]
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        assert self.total_num_kv_heads % tp_size == 0
        self.num_kv_heads = self.total_num_kv_heads // tp_size
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5
        self.qkv_bias = qkv_bias

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
        )
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=rope_theta,
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
        )
        if not self.qkv_bias:
            self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        if not self.qkv_bias:
            q = self.q_norm(q)
            k = self.k_norm(k)
        q, k = self.rotary_emb(positions, q, k)
        o = self.attn(q, k, v)
        output = self.o_proj(o.flatten(1, -1))
        return output




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
