# -*- coding: utf-8 -*-

# ***************************************************
# * File        : sparse_attention.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2026-10-24
# * Version     : 1.0.102421
# * Description : DeepSeek-V3 Native Sparse Attention（NSA）教学实现。
# *               三分支（均因果）：
# *                 1. compressed：K/V 按块平均池化后做块级注意力（全局粗上下文）；
# *                 2. selected：按 compressed 分数对每个 query 选 top-N 个
# *                    历史完整块做精确注意力；
# *                 3. sliding window：最近 window 个 token 的精确注意力。
# *               输出 = σ(g_c)·O_c + σ(g_s)·O_s + σ(g_w)·O_w（每头可学习门控，
# *               论文式(5)，arXiv:2412.19437 §2.2）。
# *               教学简化：单 kernel 稠密实现（不分解 FlashAttention kernel）；
# *               selected 与 window 的覆盖允许重叠（论文中互斥）；
# *               当前不完整块只由 window 分支覆盖（与论文一致）。
# * Link        : https://arxiv.org/abs/2412.19437
# * Requirement : torch
# ***************************************************

__all__ = ["NativeSparseAttention"]

# python libraries
from pathlib import Path

import torch
import torch.nn as nn

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


class NativeSparseAttention(nn.Module):
    """NSA（DeepSeek-V3 §2.2）教学版：compressed + selected + sliding window 三分支门控融合。

    Args:
        d_model: 模型隐藏维
        n_heads: 注意力头数
        head_dim: 每头维数（缺省 d_model // n_heads）
        context_length: 位置编码无关，仅作文档约定/上限提示
        block_size: compressed/selected 的块宽 w
        n_selected_blocks: selected 分支每 query 选取的历史块数
        sliding_window: window 分支的窗口大小
    """

    def __init__(self, d_model, n_heads, head_dim=None, context_length=1024,
                 block_size=16, n_selected_blocks=2, sliding_window=32, dtype=None):
        super().__init__()

        self.n_heads = n_heads
        self.head_dim = head_dim if head_dim is not None else d_model // n_heads
        self.d_out = self.n_heads * self.head_dim
        self.block_size = block_size
        self.n_selected_blocks = n_selected_blocks
        self.sliding_window = sliding_window
        # projections（DeepSeek 无注意力 bias）
        self.W_query = nn.Linear(d_model, self.d_out, bias=False, dtype=dtype)
        self.W_key = nn.Linear(d_model, self.d_out, bias=False, dtype=dtype)
        self.W_value = nn.Linear(d_model, self.d_out, bias=False, dtype=dtype)
        self.out_proj = nn.Linear(self.d_out, d_model, bias=False, dtype=dtype)
        # 每头门控（sigmoid 后为分支权重）
        self.gate_compressed = nn.Parameter(torch.zeros(n_heads))
        self.gate_selected = nn.Parameter(torch.zeros(n_heads))
        self.gate_window = nn.Parameter(torch.zeros(n_heads))

    # ------------------------------
    # 工具：块级因果可见性
    # ------------------------------
    def _block_causal(self, num_tokens, num_blocks, device):
        """query t 可见完整块 b 的条件：(b+1)*w <= t（当前不完整块不可见）。

        Returns:
            (num_tokens, num_blocks) bool，True=可见
        """
        t = torch.arange(num_tokens, device=device)[:, None]
        b = torch.arange(num_blocks, device=device)[None, :]
        return ((b + 1) * self.block_size) <= t

    def _pool_blocks(self, k, v):
        """K/V 按块平均池化：尾部零填充到块宽整数倍（填充位不可见，见 _block_causal）。

        Args:
            k/v: (B, H, T, D)

        Returns:
            k_blk/v_blk: (B, H, num_blocks, D)
        """
        w = self.block_size
        num_tokens = k.shape[2]
        pad = (-num_tokens) % w
        if pad:
            k = torch.nn.functional.pad(k, (0, 0, 0, pad))
            v = torch.nn.functional.pad(v, (0, 0, 0, pad))
        num_blocks = k.shape[2] // w
        k_blk = k.reshape(*k.shape[:2], num_blocks, w, k.shape[-1]).mean(dim=3)
        v_blk = v.reshape(*v.shape[:2], num_blocks, w, v.shape[-1]).mean(dim=3)
        return k_blk, v_blk

    # ------------------------------
    # 分支 1：compressed（块级粗注意力）
    # ------------------------------
    def _compressed_out(self, q, k, v, block_causal):
        """q: (B,H,T,D)；返回 (O_c, scores_c)：O_c (B,H,T,D)，scores_c (B,H,T,Nb)。"""
        k_blk, v_blk = self._pool_blocks(k, v)
        scores_c = q @ k_blk.transpose(-2, -1) / self.head_dim**0.5
        scores_c = scores_c.masked_fill(~block_causal, -torch.inf)
        attn_c = torch.softmax(scores_c, dim=-1)
        # t < w 的 query 无可见块（整行 -inf -> NaN），该分支输出 0
        attn_c = torch.nan_to_num(attn_c, nan=0.0)
        return attn_c @ v_blk, scores_c

    # ------------------------------
    # 分支 2：selected（每 query top-N 块精确注意力）
    # ------------------------------
    def _selected_out(self, q, k, v, scores_c, block_causal):
        B, H, T, D = q.shape
        w = self.block_size
        # 块级重要性 = causally-masked 的 compressed 分数
        n_sel = min(self.n_selected_blocks, scores_c.shape[-1])
        top_idx = scores_c.topk(n_sel, dim=-1).indices                    # (B,H,T,N)
        # 精确块 KV（含尾部 padding 的最后块，靠 valid 掩码挡掉）
        pad = (-T) % w
        k_pad = torch.nn.functional.pad(k, (0, 0, 0, pad)) if pad else k
        v_pad = torch.nn.functional.pad(v, (0, 0, 0, pad)) if pad else v
        num_blocks = k_pad.shape[2] // w
        k_blk = k_pad.reshape(B, H, num_blocks, w, D)
        v_blk = v_pad.reshape(B, H, num_blocks, w, D)
        # gather：按 top_idx 取每个 query 的 N 个块 -> (B,H,T,N,w,D)
        idx = top_idx[..., None, None].expand(-1, -1, -1, -1, w, D)
        k_sel = torch.gather(
            k_blk.unsqueeze(2).expand(-1, -1, T, -1, -1, -1), 3, idx)
        v_sel = torch.gather(
            v_blk.unsqueeze(2).expand(-1, -1, T, -1, -1, -1), 3, idx)
        # 对选中的 N*w 个 token 做（联合）softmax
        scores_s = torch.einsum("bhtd,bhtnwd->bhtnw", q, k_sel) / self.head_dim**0.5
        valid = torch.gather(
            block_causal[None, None].expand(B, H, T, -1), 3, top_idx)    # (B,H,T,N)
        scores_s = scores_s.masked_fill(~valid[..., None], -torch.inf)
        # 联合 softmax（torch.softmax 不支持 tuple dim：展平 N*w 再还原）
        B_, H_, T_, N_, W_ = scores_s.shape
        attn_s = torch.softmax(scores_s.reshape(B_, H_, T_, N_ * W_), dim=-1)
        attn_s = torch.nan_to_num(attn_s, nan=0.0).reshape(B_, H_, T_, N_, W_)
        return torch.einsum("bhtnw,bhtnwd->bhtd", attn_s, v_sel)

    # ------------------------------
    # 分支 3：sliding window（精确因果窗口注意力）
    # ------------------------------
    def _window_out(self, q, k, v):
        T = q.shape[2]
        scores_w = q @ k.transpose(-2, -1) / self.head_dim**0.5          # (B,H,T,T)
        t = torch.arange(T, device=q.device)[:, None]
        j = torch.arange(T, device=q.device)[None, :]
        window_causal = (j <= t) & (j > t - self.sliding_window)         # (T,T)
        scores_w = scores_w.masked_fill(~window_causal, -torch.inf)
        attn_w = torch.softmax(scores_w, dim=-1)
        return attn_w @ v

    def forward(self, x):
        """x: (B, T, d_model) -> (B, T, d_model)。"""
        batch_size, num_tokens, _ = x.shape
        H, D = self.n_heads, self.head_dim
        q = self.W_query(x).view(batch_size, num_tokens, H, D).transpose(1, 2)
        k = self.W_key(x).view(batch_size, num_tokens, H, D).transpose(1, 2)
        v = self.W_value(x).view(batch_size, num_tokens, H, D).transpose(1, 2)
        # 块级因果掩码
        num_blocks = (num_tokens + self.block_size - 1) // self.block_size
        block_causal = self._block_causal(num_tokens, num_blocks, x.device)
        # 三分支
        out_c, scores_c = self._compressed_out(q, k, v, block_causal)
        out_s = self._selected_out(q, k, v, scores_c, block_causal)
        out_w = self._window_out(q, k, v)
        # 每头门控融合：sigmoid -> (1,H,1,1)
        g_c = torch.sigmoid(self.gate_compressed).view(1, H, 1, 1)
        g_s = torch.sigmoid(self.gate_selected).view(1, H, 1, 1)
        g_w = torch.sigmoid(self.gate_window).view(1, H, 1, 1)
        out = g_c * out_c + g_s * out_s + g_w * out_w
        # merge heads -> out proj
        out = out.transpose(1, 2).contiguous().view(batch_size, num_tokens, self.d_out)
        return self.out_proj(out)


# 测试代码 main 函数（规范测试见 tests/unit/test_nsa.py）
def main():
    from utils.log_util import logger

    torch.manual_seed(123)
    attn = NativeSparseAttention(
        d_model=64, n_heads=4, block_size=8, n_selected_blocks=2, sliding_window=16)
    x = torch.randn(2, 32, 64)
    out = attn(x)
    logger.info(f"forward: {tuple(out.shape)}")

if __name__ == "__main__":
    main()
