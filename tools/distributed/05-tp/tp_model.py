# -*- coding: utf-8 -*-

# ***************************************************
# * File        : tp_model.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2026-10-24
# * Version     : 1.0.102403
# * Description : 05 单元：把 _common 的 toy Transformer 改写为张量并行版。
# *               - Attention：QKV 用 ColumnParallelLinear（按 head 切分），
# *                 每个 rank 只算 n_heads/p 个头；wo 用 RowParallelLinear。
# *               - FFN：w1 Column 并行 + w2 Row 并行（经典 Megatron MLP
# *                 结构），GELU 在分片上本地计算，中间无任何通信。
# *               - Embedding / 输出头 / LayerNorm：保持复制（不切分）。
# *                 toy 规模下 vocab/dim 太小，vocabulary parallel 的
# *                 all-reduce 开销大于收益；且输出头复制后 logits 天然
# *                 完整，可直接算交叉熵，便于数值等价验证。
# *               结构上每个 rank 每个 transformer block 恰好 2 次
# *               all-reduce（wo 与 w2 各一次），与 Megatron-LM 论文一致。
# *               process_group 参数（默认 None = 全局组）使切分与通信
# *               限定在 TP 子组内，供 07-compose 的 TP×DP 组合复用；
# *               目录名 05-tp 不是合法 Python 包名，兄弟模块（
# *               parallel_linear）按脚本方式直接导入（同 04-fsdp 风格）。
# * Link        : https://arxiv.org/abs/1909.08053 (Megatron-LM)
# * Requirement : torch >= 2.5；gloo（CPU）或 nccl（CUDA）后端
# ***************************************************

__all__ = ["TPAttention", "TPFeedForward", "TPTransformerBlock", "TPTransformer"]

# python libraries
import sys
from pathlib import Path

# repo root: tools/distributed/05-tp/tp_model.py -> parents[3]
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F

from parallel_linear import (
    ColumnParallelLinear,
    RowParallelLinear,
    _rank,
    _world_size,
)
from tools.distributed._common.model import ModelArgs, Transformer


class TPAttention(nn.Module):
    """
    张量并行注意力（结构对齐 _common.model.Attention）。

    切分方式：n_heads 个头均分到各 rank，rank r 只持有并计算
    n_heads/p 个头的 Q/K/V 投影与注意力。由于 _common 的 QKV 权重
    按 (head_0, head_1, ...) 连续排列，Column 并行的等长行切片恰好
    对应整头切分——每个 rank 的注意力完全独立，零通信。

    通信点：仅 wo（Row 并行）处一次 all-reduce，把各 rank 部分头
    贡献的加和汇总为完整输出。

    process_group：TP 切分/通信所依据的进程组（None = 全局组）。
    """

    def __init__(self, args: ModelArgs, process_group=None):
        super().__init__()
        world_size = _world_size(process_group)
        assert args.dim % args.n_heads == 0
        assert args.n_heads % world_size == 0, (
            f"TPAttention: n_heads={args.n_heads} 不能被 world_size={world_size} "
            f"整除（无法按整头切分）"
        )
        self.head_dim = args.dim // args.n_heads
        self.n_heads = args.n_heads
        self.n_heads_local = args.n_heads // world_size
        self.dropout_p = args.dropout_p
        self.resid_dropout = nn.Dropout(args.dropout_p)

        # QKV：Column 并行，每个 rank 只持有 n_heads_local 个头的权重行
        self.wq = ColumnParallelLinear(args.dim, args.dim, bias=False, process_group=process_group)
        self.wk = ColumnParallelLinear(args.dim, args.dim, bias=False, process_group=process_group)
        self.wv = ColumnParallelLinear(args.dim, args.dim, bias=False, process_group=process_group)
        # wo：Row 并行，输入维度与 QKV 的输出切片一致（本 rank 的若干个头）
        self.wo = RowParallelLinear(args.dim, args.dim, bias=False, process_group=process_group)

    def forward(self, x):
        bsz, seq_len, _ = x.size()
        # 本地 matmul：得到本 rank 的 n_heads_local 个头的 Q/K/V，无通信
        queries, keys, values = self.wq(x), self.wk(x), self.wv(x)
        queries = queries.view(bsz, seq_len, self.n_heads_local, self.head_dim)
        keys = keys.view(bsz, seq_len, self.n_heads_local, self.head_dim)
        values = values.view(bsz, seq_len, self.n_heads_local, self.head_dim)

        queries = queries.transpose(1, 2)  # (bsz, n_heads_local, seq_len, head_dim)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        # 每个 head 的注意力只依赖该 head 自己的 Q/K/V：本地 SDPA 即正确，无通信
        output = F.scaled_dot_product_attention(
            queries,
            keys,
            values,
            None,
            self.dropout_p if self.training else 0,
        )
        output = output.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        # wo（Row 并行）：内部完成唯一一次 all-reduce
        return self.resid_dropout(self.wo(output))


class TPFeedForward(nn.Module):
    """
    张量并行 FFN（经典 Megatron MLP 结构，对齐 _common.model.FeedForward）。

    Y = w2(GELU(w1(X)))：w1 按输出维度切（Column 并行），GELU 是逐元素
    算子、可直接在分片上计算，w2 按输入维度切（Row 并行）——两层之间
    不需要任何通信，整个 FFN 只有 w2 处一次 all-reduce。
    """

    def __init__(self, dim: int, hidden_dim: int, dropout_p: float, process_group=None):
        super().__init__()
        self.w1 = ColumnParallelLinear(dim, hidden_dim, bias=True, process_group=process_group)
        self.gelu = nn.GELU()
        self.w2 = RowParallelLinear(hidden_dim, dim, bias=True, process_group=process_group)
        self.resid_dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        # w1 输出是 hidden/p 的分片 -> GELU 本地逐元素 -> w2 内部 all-reduce
        return self.resid_dropout(self.w2(self.gelu(self.w1(x))))


class TPTransformerBlock(nn.Module):
    """对齐 _common.model.TransformerBlock，仅把 attention/FFN 换成 TP 版。"""

    def __init__(self, args: ModelArgs, process_group=None):
        super().__init__()
        self.attention_norm = nn.LayerNorm(args.dim)
        self.attention = TPAttention(args, process_group=process_group)
        self.ffn_norm = nn.LayerNorm(args.dim)
        self.feed_forward = TPFeedForward(
            args.dim, hidden_dim=4 * args.dim, dropout_p=args.dropout_p,
            process_group=process_group,
        )

    def forward(self, x):
        # 残差流的输入/输出都是完整张量（Row 并行层 all-reduce 之后各 rank 一致）
        h = x + self.attention(self.attention_norm(x))
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class TPTransformer(nn.Module):
    """
    张量并行 toy Transformer，对外行为与 _common.model.Transformer 完全
    一致：输入 (bsz, seq_len) token ids，输出 (bsz, seq_len, vocab_size)
    完整 logits（输出头复制，故 logits 在每个 rank 上都是完整的）。

    参数切分：仅 QKV/wo/w1/w2 按 world_size 切分（占模型绝大部分参数）；
    tok/pos embedding、LayerNorm、输出头在每个 rank 上完整复制。由于所有
    rank 喂同样的 token 且残差流处处一致，这些复制参数的梯度天然相同，
    无需额外同步即可保持一致（相当于参数复制的数据并行退化情形）。
    """

    def __init__(self, args: ModelArgs, process_group=None):
        super().__init__()
        assert args.vocab_size is not None
        assert args.max_seq_len is not None
        self.model_args = args
        self.max_seq_len = args.max_seq_len
        self.process_group = process_group
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
        self.pos_embeddings = nn.Embedding(args.max_seq_len, args.dim)
        self.dropout = nn.Dropout(args.dropout_p)
        self.layers = nn.ModuleList()
        for _ in range(args.n_layers):
            self.layers.append(TPTransformerBlock(args, process_group=process_group))
        self.norm = nn.LayerNorm(args.dim)
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)

    def forward(self, tokens):
        _bsz, seq_len = tokens.size()
        assert seq_len <= self.max_seq_len
        h = self.tok_embeddings(tokens)
        pos = torch.arange(0, seq_len, device=tokens.device)
        p = self.pos_embeddings(pos)
        h = h + p
        h = self.dropout(h)
        for layer in self.layers:
            h = layer(h)
        h = self.norm(h)
        output = self.output(h).float()
        return output

    @torch.no_grad()
    def load_from_full(self, full: Transformer) -> None:
        """
        从完整（未切分）的 Transformer 切出本 rank 的参数分片。

        保证 TP 模型与完整模型在数学上是同一个模型——这是
        verify_equivalence.py 数值等价验证的前提。embedding/LayerNorm/
        输出头整体复制，QKV/w1 取行切片，wo/w2 取列切片。
        """
        self.tok_embeddings.weight.copy_(full.tok_embeddings.weight)
        self.pos_embeddings.weight.copy_(full.pos_embeddings.weight)
        self.norm.weight.copy_(full.norm.weight)
        self.norm.bias.copy_(full.norm.bias)
        self.output.weight.copy_(full.output.weight)
        for tp_layer, full_layer in zip(self.layers, full.layers):
            tp_layer.attention_norm.weight.copy_(full_layer.attention_norm.weight)
            tp_layer.attention_norm.bias.copy_(full_layer.attention_norm.bias)
            tp_layer.ffn_norm.weight.copy_(full_layer.ffn_norm.weight)
            tp_layer.ffn_norm.bias.copy_(full_layer.ffn_norm.bias)
            # QKV：按输出行切（Column 并行）
            tp_layer.attention.wq.load_full_weight(full_layer.attention.wq.weight)
            tp_layer.attention.wk.load_full_weight(full_layer.attention.wk.weight)
            tp_layer.attention.wv.load_full_weight(full_layer.attention.wv.weight)
            # wo：按输入列切（Row 并行）
            tp_layer.attention.wo.load_full_weight(full_layer.attention.wo.weight)
            # FFN：w1 行切，w2 列切
            tp_layer.feed_forward.w1.load_full_weight(
                full_layer.feed_forward.w1.weight, full_layer.feed_forward.w1.bias
            )
            tp_layer.feed_forward.w2.load_full_weight(
                full_layer.feed_forward.w2.weight, full_layer.feed_forward.w2.bias
            )




# 测试代码 main 函数
def main():
    """单进程自检（world_size=1 时 TP 模型应与完整模型逐位一致）。"""
    torch.manual_seed(0)
    args = ModelArgs(n_layers=2, vocab_size=8, max_seq_len=16, dim=16, n_heads=4, dropout_p=0)
    full = Transformer(args)
    tp = TPTransformer(args)
    tp.load_from_full(full)
    full.eval()
    tp.eval()
    tokens = torch.randint(0, args.vocab_size, (2, 8))
    with torch.no_grad():
        diff = (tp(tokens) - full(tokens)).abs().max().item()
    assert diff < 1e-6, f"logits diff {diff}"
    print(f"single-process equivalence OK (max logits diff {diff:.2e}, world_size={_world_size()})")


if __name__ == "__main__":
    main()
