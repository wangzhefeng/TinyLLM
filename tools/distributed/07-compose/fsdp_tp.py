# -*- coding: utf-8 -*-

# ***************************************************
# * File        : fsdp_tp.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2026-10-24
# * Version     : 1.0.102401
# * Description : 07 单元（附属）：工业形态的 FSDP2 + TP 组合并行示例
# *               （torchtitan 风格）。与 main.py 手写版的对照关系：
# *               手写版用 dist.new_group 显式划分 TP/DP 通信子组、
# *               手写 DP 梯度 all-reduce，让每一步通信完全可见；
# *               工业版用 init_device_mesh("cuda", (dp, tp)) 一次声明
# *               二维网格，TP 计划（ColwiseParallel/RowwiseParallel）
# *               与 fully_shard(mesh=dp_mesh) 各自认领一个网格维度，
# *               通信组划分与梯度同步全由 device_mesh / FSDP2 代劳。
# *               CUDA-only：fully_shard 依赖 CUDA stream（同 04-fsdp），
# *               无 GPU 时入口明确报错退出；TP 的 DTensor 路径在 CPU
# *               上虽有部分支持，但组合形态的生产语义就是 GPU 集群。
# * Link        : https://github.com/pytorch/torchtitan/blob/main/docs/fsdp.md
# * Requirement : torch >= 2.5；CUDA GPU + nccl
# ***************************************************

# python libraries
import argparse
import os
import sys
from pathlib import Path

# repo root: tools/distributed/07-compose/fsdp_tp.py -> parents[3]
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import torch
import torch.distributed as dist
import torch.nn.functional as F

# FSDP2 导入：torch >= 2.6 在 torch.distributed.fsdp 下，2.5 在 _composable 下
try:
    from torch.distributed.fsdp import fully_shard
except ImportError:
    from torch.distributed._composable.fsdp import fully_shard

from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    parallelize_module,
)

from tools.distributed._common.dist import rank0_print
from tools.distributed._common.model import ModelArgs, Transformer


def apply_tp(model: Transformer, tp_mesh) -> None:
    """
    对每个 transformer block 应用 TP 计划（torchtitan 同款切法）：

      - attention.wq/wk/wv：ColwiseParallel（按输出维度切 = 按 head 切），
        对应手写版的 ColumnParallelLinear；
      - attention.wo：RowwiseParallel（按输入维度切，输出自动 all-reduce），
        对应手写版的 RowParallelLinear；
      - feed_forward.w1：ColwiseParallel；feed_forward.w2：RowwiseParallel，
        即经典 Megatron MLP 结构。

    与手写版（05-tp/parallel_linear.py + 07-compose/main.py）的差异：
    手写版的 f/g autograd 原语在 TP 子组内显式 all-reduce；这里的
    Colwise/Rowwise 计划内部用 DTensor 完成同样的通信，通信组由
    tp_mesh 隐式指定——mesh 的一维就对应手写版的一个 new_group。
    """
    for layer in model.layers:
        plan = {
            "attention.wq": ColwiseParallel(),
            "attention.wk": ColwiseParallel(),
            "attention.wv": ColwiseParallel(),
            "attention.wo": RowwiseParallel(),
            "feed_forward.w1": ColwiseParallel(),
            "feed_forward.w2": RowwiseParallel(),
        }
        parallelize_module(layer, tp_mesh, plan)


def apply_fsdp(model: Transformer, dp_mesh) -> None:
    """
    在 DP 网格维度上逐层 fully_shard：TP 切完之后的"每个 TP rank 持有的
    本地参数"再沿 DP 维度分片（ZeRO-3 语义），梯度同步（reduce-scatter）
    由 FSDP2 在 dp_mesh 上自动完成——对应手写版 sync_dp_grads 的
    all-reduce，但进一步省了优化器状态与梯度显存。
    """
    for layer in model.layers:
        fully_shard(layer, mesh=dp_mesh)
    fully_shard(model, mesh=dp_mesh)


def main():
    parser = argparse.ArgumentParser(description="07-compose: FSDP2 + TP (torchtitan style, CUDA-only)")
    parser.add_argument("--tp-size", type=int, default=2)
    parser.add_argument("--dp-size", type=int, default=None, help="默认 world_size/tp_size")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--dim", type=int, default=16)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--vocab-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-2)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--log-every", type=int, default=1)
    args = parser.parse_args()

    # FSDP2（fully_shard）依赖 CUDA stream 做通信重叠，CPU + gloo 下直接
    # 报 "FSDP requires CUDA for streams"（同 04-fsdp）：本示例为 CUDA-only，
    # 入口处明确报错，不做隐晦的 CPU 回退。CPU 上的组合并行教学见 main.py。
    if not torch.cuda.is_available():
        raise SystemExit(
            "07-compose/fsdp_tp.py 需要 CUDA GPU：FSDP2(fully_shard) 依赖 CUDA stream，"
            "不支持 CPU/MPS。手写 TP×DP 组合（CPU+gloo 可运行）请改用 "
            "tools/distributed/07-compose/main.py。"
        )

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    dist.init_process_group(backend="nccl", device_id=device)
    world_size = dist.get_world_size()

    dp_size = args.dp_size if args.dp_size is not None else world_size // args.tp_size
    assert args.tp_size * dp_size == world_size, (
        f"tp_size({args.tp_size}) x dp_size({dp_size}) != world_size({world_size})"
    )

    # 一次声明二维网格：mesh["tp"] / mesh["dp"] 即手写版里的两类 new_group。
    # 默认布局与手写版一致——rank = dp_rank * tp_size + tp_rank（TP 最内层，
    # 连续 rank 同组），TP 组天然落在同一节点内。
    mesh = init_device_mesh(
        "cuda", (dp_size, args.tp_size), mesh_dim_names=("dp", "tp")
    )
    rank0_print(f"[fsdp+tp] world_size={world_size} = dp({dp_size}) x tp({args.tp_size}), mesh={mesh}")

    torch.manual_seed(args.seed)
    model_args = ModelArgs(
        n_layers=args.n_layers,
        vocab_size=args.vocab_size,
        max_seq_len=args.seq_len,
        dim=args.dim,
        n_heads=args.n_heads,
        dropout_p=0,  # TP 分片间 dropout RNG 不对齐（同手写版）
    )
    with torch.device("meta"):
        model = Transformer(model_args)

    # 先 TP 后 FSDP（torchtitan 顺序）：TP 把参数切成 DTensor 分片，
    # FSDP 再在这些分片上沿 DP 维度进一步切。
    apply_tp(model, mesh["tp"])
    apply_fsdp(model, mesh["dp"])
    model.to_empty(device=device)
    model.reset_parameters()

    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    for step in range(args.steps):
        # 各 DP 组喂不同随机数据（TP 组内 seed 相同 -> 数据一致）：
        # 每个 DP rank 用 (seed + dp 坐标) 派生自己的数据流即可，这里
        # 用 mesh 坐标偏移演示，教学重点在并行结构本身。
        dp_rank = mesh["dp"].get_local_rank()
        data_gen = torch.Generator(device=device).manual_seed(args.seed + 1 + dp_rank)
        tokens = torch.randint(
            0, args.vocab_size, (args.batch_size, args.seq_len),
            generator=data_gen, device=device,
        )
        targets = torch.randint(
            0, args.vocab_size, (args.batch_size, args.seq_len),
            generator=data_gen, device=device,
        )
        logits = model(tokens)
        loss = F.cross_entropy(logits.view(-1, args.vocab_size), targets.view(-1))
        loss.backward()  # DP 梯度同步由 FSDP2 自动完成，无需手写
        optim.step()
        optim.zero_grad()
        if step % args.log_every == 0:
            rank0_print(f"step {step:>4d} | loss {loss.item():.4f}")

    rank0_print("[fsdp+tp] done")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
