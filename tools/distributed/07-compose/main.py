# -*- coding: utf-8 -*-

# ***************************************************
# * File        : main.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2026-10-24
# * Version     : 1.0.102401
# * Description : 07 单元：组合并行 —— 手写 TP×DP 二维组合（torchrun 启动，
# *               CPU+gloo 可运行）。真实大模型训练从来不是单一并行模式：
# *               张量并行（TP）通信密集，放在节点内（NVLink）；数据并行
# *               （DP）通信稀疏（每步一次梯度同步），放在节点间（IB/RoCE）。
# *               本脚本用 4 个 CPU 进程组织成 2×2 网格演示这一结构：
# *                 - TP 维度：复用 05-tp 的 TPTransformer，通过
# *                   process_group 参数把 f/g 原语的通信约束在 TP 子组内；
# *                 - DP 维度：各 DP 组喂不同数据分片，反向后对全部参数
# *                   的梯度在 DP 子组内 all-reduce 取平均（手写同步，
# *                   思路与 03-zero 的 ZeRO-1 一致，区别仅在 ZeRO 把
# *                   优化器状态也切分；本单元不依赖 03 的代码）。
# *               rank0 打印网格拓扑、每 rank 参数量与吞吐语义说明。
# *               dropout_p=0：TP 分片间 dropout RNG 不对齐（同 05-tp）。
# * Link        : https://arxiv.org/abs/1909.08053 (Megatron-LM)
# * Requirement : torch >= 2.5；gloo（CPU）或 nccl（CUDA）后端
# ***************************************************

__all__ = [
    "make_parallel_groups",
    "sync_dp_grads",
    "build_models",
    "make_sharded_batches",
    "shard_batches",
    "average_loss_over_dp",
]

# python libraries
import argparse
import os
import sys
from pathlib import Path

# repo root: tools/distributed/07-compose/main.py -> parents[3]
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
# 05-tp 的模块（parallel_linear / tp_model）按脚本方式导入：
# 单元目录名不是合法 Python 包名，把 05-tp 目录加入 sys.path
TP_DIR = ROOT / "tools" / "distributed" / "05-tp"
if str(TP_DIR) not in sys.path:
    sys.path.append(str(TP_DIR))

import torch
import torch.distributed as dist
import torch.nn.functional as F

from tp_model import TPTransformer
from tools.distributed._common.dist import cleanup, rank0_print, setup
from tools.distributed._common.metrics import log_step_metrics
from tools.distributed._common.model import ModelArgs, Transformer


def make_parallel_groups(world_size: int, tp_size: int):
    """
    把 world_size 个 rank 组织成 dp_size × tp_size 的二维网格。

    rank 布局（Megatron / torchtitan 约定，TP 为最内层维度）：

        rank = dp_rank * tp_size + tp_rank

    即连续的 tp_size 个 rank 组成一个 TP 组。torchrun 按节点顺序分配
    rank（同一节点上的 LOCAL_RANK 连续），所以只要 tp_size <= 每节点
    进程数，每个 TP 组就天然落在同一节点内——这正是"TP 走节点内
    NVLink、DP 走节点间"的工程对应物。

    易错点（教学重点）：
      1. dist.new_group 是集合操作：所有 rank 必须对所有组都调用一次
         new_group（即使自己不在组内），否则挂起或错位；
      2. TP 的 f/g 通信必须限定在 TP 子组内（parallel_linear 的
         process_group 参数）。若误用全局组，不同 DP 组的梯度会被
         错误地加在一起，loss 会"看起来在降"但数学上完全错误；
      3. TP 组内所有 rank 必须喂同样的数据（否则各分片算的就不是
         同一个逻辑 batch，等价性被破坏）；数据分片只沿 DP 维度变化。

    Returns:
        (tp_group, dp_group, tp_rank, dp_rank, tp_groups, dp_groups)
        tp_groups/dp_groups 为各组的 rank 列表（供拓扑打印）。
    """
    assert world_size % tp_size == 0, (
        f"world_size={world_size} 不能被 tp_size={tp_size} 整除"
    )
    dp_size = world_size // tp_size
    rank = dist.get_rank()

    tp_group = dp_group = None
    tp_rank = dp_rank = None
    # TP 组：连续的 tp_size 个 rank（对应同一节点内的 GPU）
    tp_groups = [list(range(d * tp_size, (d + 1) * tp_size)) for d in range(dp_size)]
    # DP 组：每个 TP 组中同位置的 rank（对应不同节点上同槽位的 GPU）
    dp_groups = [[d * tp_size + t for d in range(dp_size)] for t in range(tp_size)]
    for ranks in tp_groups:
        group = dist.new_group(ranks=ranks)
        if rank in ranks:
            tp_group = group
            tp_rank = ranks.index(rank)
    for ranks in dp_groups:
        group = dist.new_group(ranks=ranks)
        if rank in ranks:
            dp_group = group
            dp_rank = ranks.index(rank)
    return tp_group, dp_group, tp_rank, dp_rank, tp_groups, dp_groups


def sync_dp_grads(model: torch.nn.Module, dp_group, dp_size: int) -> None:
    """
    手写 DP 梯度同步：反向后对所有参数的梯度在 DP 组内 all-reduce 取平均。

    为什么"所有参数"一把梭就是对的：
      - 复制参数（embedding / LayerNorm / 输出头）：DP 组内每个 rank 持有
        完整副本，all-reduce 平均就是标准的数据并行梯度平均；
      - TP 分片参数（QKV/wo/w1/w2）：DP 组的成员是各 TP 组中"同槽位"的
        rank（见 make_parallel_groups 的布局），它们持有的分片形状相同、
        语义对应，逐元素 all-reduce 恰好把"同一分片、不同数据"的梯度平均。
    这和 03-zero 的 ZeRO-1 是同一个 all-reduce，区别只在 ZeRO 进一步
    切分优化器状态；本单元刻意手写，让通信组完全显式可见。
    """
    if dp_size == 1:
        return
    for p in model.parameters():
        if p.grad is None:
            continue
        dist.all_reduce(p.grad, group=dp_group)
        p.grad.div_(dp_size)


def average_loss_over_dp(loss: torch.Tensor, dp_group, dp_size: int) -> float:
    """把各 DP 组的本地 loss（各自数据分片上的均值）平均成全局 batch 的 loss。"""
    if dp_size == 1:
        return loss.item()
    loss_tensor = loss.detach().clone()
    dist.all_reduce(loss_tensor, group=dp_group)
    return (loss_tensor / dp_size).item()


def build_models(model_args: ModelArgs, seed: int, tp_group) -> tuple[Transformer, TPTransformer]:
    """
    固定 seed 构造完整模型（所有 rank 上逐位相同），再切分出本 rank 的
    TP 分片（切分与通信都限定在 tp_group 内）——保证组合模型与完整模型
    在数学上是同一个模型，这是 verify_equivalence.py 的前提。
    """
    torch.manual_seed(seed)
    full = Transformer(model_args)
    tp = TPTransformer(model_args, process_group=tp_group)
    tp.load_from_full(full)
    return full, tp


def shard_batches(
    model_args: ModelArgs,
    batch_size: int,
    dp_size: int,
    num_batches: int,
    seed: int,
) -> list[list[tuple[torch.Tensor, torch.Tensor]]]:
    """
    生成固定的合成批次，每个 batch 沿 DP 维度切成 dp_size 个分片：
    batches[step][dp_rank] = (tokens, targets)。

    生成器 seed 固定，所有 rank 生成出完全相同的全量数据，各自只消费
    自己 dp_rank 对应的分片——不需要任何数据广播。DP rank d 在 step s
    消费的分片，与单进程参考实现中第 s 个有效 batch 的第 d 段完全一致
    （见 verify_equivalence.py 的对齐方式）。
    """
    gen = torch.Generator().manual_seed(seed + 1)
    batches = []
    for _ in range(num_batches):
        shards = []
        for _ in range(dp_size):
            tokens = torch.randint(
                0, model_args.vocab_size, (batch_size, model_args.max_seq_len), generator=gen
            )
            targets = torch.randint(
                0, model_args.vocab_size, (batch_size, model_args.max_seq_len), generator=gen
            )
            shards.append((tokens, targets))
        batches.append(shards)
    return batches


def report_topology(rank: int, world_size: int, tp_size: int, dp_size: int,
                    tp_groups, dp_groups, tp_rank: int, dp_rank: int) -> None:
    """rank0 打印 2D 网格拓扑；每个 rank 打印自己的坐标。"""
    rank0_print(f"[grid] world_size={world_size} = dp_size({dp_size}) x tp_size({tp_size})")
    rank0_print("[grid] rank 布局: rank = dp_rank * tp_size + tp_rank（TP 为最内层维度）")
    for d, ranks in enumerate(tp_groups):
        rank0_print(f"[grid]   TP 组 {d}（节点内，NVLink 语义）: ranks {ranks}")
    for t, ranks in enumerate(dp_groups):
        rank0_print(f"[grid]   DP 组 {t}（节点间，IB/RoCE 语义）: ranks {ranks}")
    print(f"[grid]   rank {rank}: dp_rank={dp_rank}, tp_rank={tp_rank}", flush=True)


def report_param_counts(tp: TPTransformer, full_numel: int, rank: int, tp_size: int) -> None:
    """收集并打印每 rank 实际持有的参数量（TP 切分后）。"""
    local_numel = sum(p.numel() for p in tp.parameters())
    counts = [None] * dist.get_world_size()
    dist.all_gather_object(counts, (rank, local_numel))
    if rank == 0:
        print(f"[params] 完整模型参数量: {full_numel}")
        for r, n in counts:
            print(f"[params]   rank {r}: {n} 参数（{n / full_numel:.2f} x 完整模型）")
        print(
            f"[params] 每个 rank 的参数约为完整模型的 1/tp_size={tp_size}（QKV/wo/w1/w2 切片）"
            " + 复制参数（embedding/LayerNorm/输出头）"
        )


def report_throughput_semantics(batch_size: int, seq_len: int, dp_size: int, tp_size: int) -> None:
    """rank0 打印与单进程等效的吞吐语义说明。"""
    rank0_print(
        f"[吞吐语义] 每步有效全局 batch = batch_size({batch_size}) x dp_size({dp_size})"
        f" = {batch_size * dp_size} 条序列"
        f"（{batch_size * dp_size * seq_len} tokens/step），与单进程跑"
        f" batch_size={batch_size * dp_size} 完全等效。"
    )
    rank0_print(
        f"[吞吐语义] TP 组内 {tp_size} 个 rank 处理同一份数据、各算 1/{tp_size} 的矩阵列/行，"
        "通信发生在每层 wo/w2 的 all-reduce（节点内，延迟敏感）；"
        f"DP 组间每步只做一次梯度 all-reduce（{dp_size} 路平均，节点间，带宽敏感）。"
    )
    rank0_print(
        "[吞吐语义] 本机 4 个 CPU 进程只是结构演示：真实集群中 TP 组应落在同一节点"
        "（torchrun 的连续 rank 即同节点），DP 组跨节点扩展数据吞吐。"
    )


def main():
    parser = argparse.ArgumentParser(description="07-compose: hand-written TP x DP 2D parallel training")
    parser.add_argument("--tp-size", type=int, default=2, help="张量并行度（节点内）")
    parser.add_argument("--dp-size", type=int, default=None, help="数据并行度（节点间），默认 world_size/tp_size")
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=8, help="每个 DP 组的本地 batch size")
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
    os.environ.setdefault("MASTER_PORT", "29516")
    rank, world_size, local_rank = setup()

    dp_size = args.dp_size if args.dp_size is not None else world_size // args.tp_size
    assert args.tp_size * dp_size == world_size, (
        f"tp_size({args.tp_size}) x dp_size({dp_size}) != world_size({world_size})"
    )
    tp_group, dp_group, tp_rank, dp_rank, tp_groups, dp_groups = make_parallel_groups(
        world_size, args.tp_size
    )
    report_topology(rank, world_size, args.tp_size, dp_size, tp_groups, dp_groups, tp_rank, dp_rank)

    model_args = ModelArgs(
        n_layers=args.n_layers,
        vocab_size=args.vocab_size,
        max_seq_len=args.seq_len,
        dim=args.dim,
        n_heads=args.n_heads,
        dropout_p=0,  # TP 分片间 dropout RNG 不对齐（同 05-tp）
    )
    full, tp = build_models(model_args, args.seed, tp_group)
    report_param_counts(tp, sum(p.numel() for p in full.parameters()), rank, args.tp_size)
    report_throughput_semantics(args.batch_size, args.seq_len, dp_size, args.tp_size)
    del full  # 完整模型仅用于参数对照，训练只用 TP 分片模型

    # 固定数据：反复在同一批数据上训练使 loss 稳定下降
    batches = shard_batches(model_args, args.batch_size, dp_size, num_batches=4, seed=args.seed)
    tp.train()
    optim = torch.optim.Adam(tp.parameters(), lr=args.lr)

    for step in range(args.steps):
        # TP 组内所有 rank 消费同一分片；数据分片只沿 DP 维度变化
        tokens, targets = batches[step % len(batches)][dp_rank]
        logits = tp(tokens)
        loss = F.cross_entropy(logits.view(-1, args.vocab_size), targets.view(-1))
        loss.backward()
        # DP 维度的全部工作就这一行：梯度在 DP 组内 all-reduce 平均
        sync_dp_grads(tp, dp_group, dp_size)
        optim.step()
        optim.zero_grad()
        # 各 DP 组的 loss 平均 = 全局有效 batch 的 loss
        global_loss = average_loss_over_dp(loss, dp_group, dp_size)
        if step % args.log_every == 0:
            log_step_metrics(step, global_loss)

    rank0_print(f"final loss {global_loss:.4f}（tp={args.tp_size} x dp={dp_size} 训练完成）")
    cleanup()


if __name__ == "__main__":
    main()
