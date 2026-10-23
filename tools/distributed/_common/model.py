# -*- coding: utf-8 -*-

# ***************************************************
# * File        : model.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-08-20
# * Version     : 1.0.082003
# * Description : 分布式示例共用的 toy 模型：
# *               - ToyMLP：小型 MLP 分类器，配 _common/dataset.py 的 toy
# *                 数据集，用于 00/01/02/03 单元（单卡/DP/DDP/ZeRO）。
# *               - Transformer + ModelArgs：toy Transformer（pre-norm、
# *                 SDPA 注意力、GELU FFN），用于 04/06/07 单元
# *                 （FSDP/PP/组合策略），迁移自原 fsdp/model.py。
# * Link        : https://github.com/pytorch/examples/tree/main/distributed/FSDP2
# * Requirement : torch >= 2.5
# ***************************************************

__all__ = ["ToyMLP", "ModelArgs", "Transformer"]

# python libraries
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


class ToyMLP(nn.Module):
    """
    小型 MLP 分类器，输入/输出维度与 _common/dataset.py 的 toy 数据对齐。
    """

    def __init__(self, num_features: int = 20, num_classes: int = 2, hidden_sizes=(64, 32)):
        super().__init__()
        layers = []
        in_dim = num_features
        for hidden in hidden_sizes:
            layers += [nn.Linear(in_dim, hidden), nn.ReLU()]
            in_dim = hidden
        layers.append(nn.Linear(in_dim, num_classes))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


@dataclass
class ModelArgs:
    n_layers: int = 2
    vocab_size: int = 8
    max_seq_len: int = 16
    dim: int = 16
    n_heads: int = 4
    dropout_p: float = 0.1


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        assert args.dim % args.n_heads == 0
        self.head_dim = args.dim // args.n_heads
        self.n_heads = args.n_heads
        self.dropout_p = args.dropout_p
        self.resid_dropout = nn.Dropout(args.dropout_p)

        self.wq = nn.Linear(args.dim, args.dim, bias=False)
        self.wk = nn.Linear(args.dim, args.dim, bias=False)
        self.wv = nn.Linear(args.dim, args.dim, bias=False)
        self.wo = nn.Linear(args.dim, args.dim, bias=False)

    def forward(self, x):
        bsz, seq_len, _ = x.size()
        queries, keys, values = self.wq(x), self.wk(x), self.wv(x)
        queries = queries.view(bsz, seq_len, self.n_heads, self.head_dim)
        keys = keys.view(bsz, seq_len, self.n_heads, self.head_dim)
        values = values.view(bsz, seq_len, self.n_heads, self.head_dim)

        queries = queries.transpose(1, 2)  # (bsz, n_heads, seq_len, head_dim)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        output = F.scaled_dot_product_attention(
            queries,
            keys,
            values,
            None,
            self.dropout_p if self.training else 0,
        )
        output = output.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        return self.resid_dropout(self.wo(output))

    def reset_parameters(self):
        self.wq.reset_parameters()
        self.wk.reset_parameters()
        self.wv.reset_parameters()
        self.wo.reset_parameters()


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout_p):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim)
        self.gelu = nn.GELU()
        self.w2 = nn.Linear(hidden_dim, dim)
        self.resid_dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        return self.resid_dropout(self.w2(self.gelu(self.w1(x))))

    def reset_parameters(self):
        self.w1.reset_parameters()
        self.w2.reset_parameters()


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.attention_norm = nn.LayerNorm(args.dim)
        self.attention = Attention(args)
        self.ffn_norm = nn.LayerNorm(args.dim)
        self.feed_forward = FeedForward(
            args.dim, hidden_dim=4 * args.dim, dropout_p=args.dropout_p
        )

    def forward(self, x):
        h = x + self.attention(self.attention_norm(x))
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

    def reset_parameters(self):
        self.attention_norm.reset_parameters()
        self.attention.reset_parameters()
        self.ffn_norm.reset_parameters()
        self.feed_forward.reset_parameters()


# A toy transformer model, partly inspired by the nanoGPT model:
# https://github.com/karpathy/nanoGPT.
class Transformer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        assert args.vocab_size is not None
        assert args.max_seq_len is not None
        self.model_args = args
        self.max_seq_len = args.max_seq_len
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
        self.pos_embeddings = nn.Embedding(args.max_seq_len, args.dim)
        self.dropout = nn.Dropout(args.dropout_p)
        self.layers = nn.ModuleList()
        for _ in range(args.n_layers):
            self.layers.append(TransformerBlock(args))
        self.norm = nn.LayerNorm(args.dim)
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)

    def forward(self, tokens):
        _bsz, seq_len = tokens.size()
        assert seq_len <= self.max_seq_len
        h = self.tok_embeddings(tokens)
        pos = torch.arange(0, seq_len, device=tokens.device)
        p = self.pos_embeddings(pos)  # positional embeddings of shape (seq_len, dim)
        h = h + p
        h = self.dropout(h)
        for layer in self.layers:
            h = layer(h)
        h = self.norm(h)
        output = self.output(h).float()
        return output

    def reset_parameters(self):
        self.tok_embeddings.reset_parameters()
        self.pos_embeddings.reset_parameters()
        self.norm.reset_parameters()
        self.output.reset_parameters()




# 测试代码 main 函数
def main():
    mlp = ToyMLP()
    print(f"ToyMLP params: {sum(p.numel() for p in mlp.parameters())}")

    args = ModelArgs(n_layers=2, n_heads=4, vocab_size=128, max_seq_len=32, dropout_p=0)
    model = Transformer(args)
    x = torch.randint(0, 128, (4, 32))
    print(f"Transformer out: {tuple(model(x).shape)}")

if __name__ == "__main__":
    main()
