# -*- coding: utf-8 -*-

# ***************************************************
# * File        : main.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2026-10-24
# * Version     : 1.0.102403
# * Description : 05 单元：张量并行（TP）toy Transformer 训练入口
# *               （torchrun 启动，CPU+gloo 可运行）。
# *               固定 seed 构造完整模型并切分到各 rank，用一批固定的
# *               合成数据训练若干步（可过拟合、loss 稳定下降）；
# *               rank0 打印逐参数的本 rank 分片 vs 完整模型的参数量
# *               对比表与总体节省比例。
# *               dropout_p=0：dropout 的 RNG 在 TP 分片间不对齐
# *               （各 rank 头数不同），本单元聚焦并行机制本身。
# * Link        : https://arxiv.org/abs/1909.08053 (Megatron-LM)
# * Requirement : torch >= 2.5；gloo（CPU）或 nccl（CUDA）后端
# ***************************************************

# python libraries
import argparse
import os
import sys
from pathlib import Path

# repo root: tools/distributed/05-tp/main.py -> parents[3]
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import torch
import torch.nn.functional as F

from parallel_linear import _rank, _world_size
from tp_model import TPTransformer
from tools.distributed._common.dist import cleanup, rank0_print, setup
from tools.distributed._common.metrics import log_step_metrics
from tools.distributed._common.model import ModelArgs, Transformer


def build_models(args) -> tuple[Transformer, TPTransformer]:
    """
    固定 seed 构造完整模型（所有 rank 上逐位相同），再切分出本 rank 的
    TP 分片——保证 TP 模型与完整模型在数学上是同一个模型。
    """
    model_args = ModelArgs(
        n_layers=args.n_layers,
        vocab_size=args.vocab_size,
        max_seq_len=args.seq_len,
        dim=args.dim,
        n_heads=args.n_heads,
        dropout_p=0,
    )
    torch.manual_seed(args.seed)
    full = Transformer(model_args)
    tp = TPTransformer(model_args)
    tp.load_from_full(full)
    return full, tp


def make_fixed_batches(args, num_batches: int = 4) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """
    固定的合成批次（seed 确定，所有 rank 完全一致）。反复在同一批数据上
    训练使 loss 可稳定下降，便于观察 TP 训练的正确性。
    """
    gen = torch.Generator().manual_seed(args.seed + 1)
    batches = []
    for _ in range(num_batches):
        tokens = torch.randint(0, args.vocab_size, (args.batch_size, args.seq_len), generator=gen)
        targets = torch.randint(0, args.vocab_size, (args.batch_size, args.seq_len), generator=gen)
        batches.append((tokens, targets))
    return batches


def report_param_sharding(full: Transformer, tp: TPTransformer) -> None:
    """rank0 打印：每个参数本 rank 实际持有量 vs 完整模型参数量。"""
    full_numels = {name: p.numel() for name, p in full.named_parameters()}
    rank0_print(f"{'parameter':<48} {'local':>8} {'full':>8} {'ratio':>7}")
    rank0_print("-" * 76)
    total_local, total_full = 0, 0
    for name, p in tp.named_parameters():
        full_numel = full_numels[name]
        total_local += p.numel()
        total_full += full_numel
        rank0_print(f"{name:<48} {p.numel():>8d} {full_numel:>8d} {p.numel()/full_numel:>7.2f}")
    rank0_print("-" * 76)
    rank0_print(f"{'TOTAL':<48} {total_local:>8d} {total_full:>8d} {total_local/total_full:>7.2f}")
    rank0_print(
        f"world_size={_world_size()}：QKV/wo/w1/w2 占模型绝大部分参数，按 rank 切分；"
        f"embedding/LayerNorm/输出头复制（toy 规模下 vocab/dim 太小，切分不划算）。"
    )


def main():
    parser = argparse.ArgumentParser(description="05-tp: Megatron-style TP toy Transformer training")
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--dim", type=int, default=16)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--vocab-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-2)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--log-every", type=int, default=4)
    args = parser.parse_args()

    # 非 torchrun 单进程直跑时补默认 rendezvous 环境变量（world_size=1）
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29514")
    rank, world_size, local_rank = setup()
    rank0_print(f"TP training: world_size={world_size}")

    full, tp = build_models(args)
    report_param_sharding(full, tp)

    batches = make_fixed_batches(args)
    tp.train()
    optim = torch.optim.Adam(tp.parameters(), lr=args.lr)

    for step in range(args.steps):
        tokens, targets = batches[step % len(batches)]
        logits = tp(tokens)
        loss = F.cross_entropy(logits.view(-1, args.vocab_size), targets.view(-1))
        loss.backward()
        optim.step()
        optim.zero_grad()
        if step % args.log_every == 0:
            log_step_metrics(step, loss.item())

    rank0_print(f"final loss {loss.item():.4f} (rank {_rank()} 训练完成)")
    cleanup()


if __name__ == "__main__":
    main()
