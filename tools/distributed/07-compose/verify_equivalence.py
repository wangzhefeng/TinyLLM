# -*- coding: utf-8 -*-

# ***************************************************
# * File        : verify_equivalence.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2026-10-24
# * Version     : 1.0.102401
# * Description : 07 单元：TP×DP 组合并行的数值等价验证（硬性自检脚本）。
# *               固定 seed 在每个 rank 上构造逐位相同的完整模型作为
# *               单进程参考（每个 rank 独立训练它，不涉及任何通信）：
# *                 1. 前向等价：同一输入下组合模型（TP 子组内通信）与
# *                    完整模型的 logits 必须在容差内一致；
# *                 2. 训练等价：数据分片对齐使"有效 batch"与单进程
# *                    完全一致（第 s 步的有效 batch = dp_size 个分片
# *                    按 dp_rank 顺序拼接），同样的优化器各训练 N 步，
# *                    全局 loss 序列必须逐步一致；训练结束后复制参数
# *                    （embedding/输出头）与参考模型逐元素对比。
# *               torchrun 4 进程（2TP×2DP）与 2 进程（1TP×2DP 或
# *               2TP×1DP）均可运行，CPU+gloo：
# *                 .venv/bin/python -m torch.distributed.run \
# *                     --nproc_per_node=4 --master_port=29515 \
# *                     tools/distributed/07-compose/verify_equivalence.py
# * Link        : https://arxiv.org/abs/1909.08053 (Megatron-LM)
# * Requirement : torch >= 2.5；gloo（CPU）或 nccl（CUDA）后端
# ***************************************************

# python libraries
import argparse
import os
import sys
from pathlib import Path

# repo root: tools/distributed/07-compose/verify_equivalence.py -> parents[3]
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
# main.py 同目录按脚本方式导入（目录名 07-compose 不是合法包名）
sys.path.append(str(Path(__file__).resolve().parent))

import torch
import torch.nn.functional as F

from main import (
    average_loss_over_dp,
    build_models,
    make_parallel_groups,
    shard_batches,
    sync_dp_grads,
)
from tools.distributed._common.dist import cleanup, rank0_print, setup
from tools.distributed._common.model import ModelArgs, Transformer

SEED = 1234
STEPS = 5
BATCH_SIZE = 4
TOL = 1e-5      # loss 序列容差（同 05-tp）
WEIGHT_TOL = 1e-4  # 训练后权重容差（Adam 多步累积浮点误差，量级大于单步 loss）


def reference_training(model_args: ModelArgs, batches, dp_size: int) -> tuple[list[float], torch.nn.Module]:
    """
    单进程参考：完整模型在"有效 batch"上训练 STEPS 步。

    有效 batch 的对齐方式：第 s 步 = batches[s][0..dp_size-1] 按 dp_rank
    顺序拼接（token 维度 cat）。交叉熵对拼接后的 batch 取均值，恰好等于
    各 DP 组本地 loss（各分片等长）的平均——这就是组合并行里的全局 loss。
    """
    torch.manual_seed(SEED)
    full = Transformer(model_args)
    full.train()
    optim = torch.optim.Adam(full.parameters(), lr=1e-2)
    losses = []
    for step in range(STEPS):
        tokens = torch.cat([batches[step][d][0] for d in range(dp_size)], dim=0)
        targets = torch.cat([batches[step][d][1] for d in range(dp_size)], dim=0)
        logits = full(tokens)
        loss = F.cross_entropy(logits.view(-1, model_args.vocab_size), targets.view(-1))
        loss.backward()
        optim.step()
        optim.zero_grad()
        losses.append(loss.item())
    return losses, full


def composed_training(model_args: ModelArgs, batches, tp_group, dp_group, dp_rank, dp_size):
    """组合并行训练：TP 子组内做张量并行，DP 组间做手写梯度同步。"""
    _full_ref, tp = build_models(model_args, SEED, tp_group)
    del _full_ref
    tp.train()
    optim = torch.optim.Adam(tp.parameters(), lr=1e-2)
    losses = []
    for step in range(STEPS):
        # TP 组内所有 rank 消费同一分片；分片只沿 DP 维度变化
        tokens, targets = batches[step][dp_rank]
        logits = tp(tokens)
        loss = F.cross_entropy(logits.view(-1, model_args.vocab_size), targets.view(-1))
        loss.backward()
        sync_dp_grads(tp, dp_group, dp_size)
        optim.step()
        optim.zero_grad()
        # 本地 loss 在 DP 组内平均 = 全局有效 batch 的 loss
        losses.append(average_loss_over_dp(loss, dp_group, dp_size))
    return losses, tp


def check_forward_equivalence(model_args: ModelArgs, batches, tp, dp_rank) -> float:
    """第 1 部分：前向 logits 等价（eval 模式）。组合模型喂本 DP 组的分片，
    参考完整模型喂同一分片（每个 rank 独立计算完整模型）。"""
    torch.manual_seed(SEED)
    full = Transformer(model_args)
    full.eval()
    tp.eval()
    tokens, _ = batches[0][dp_rank]
    with torch.no_grad():
        ref_logits = full(tokens)
        tp_logits = tp(tokens)
    max_diff = (ref_logits - tp_logits).abs().max().item()
    assert torch.allclose(tp_logits, ref_logits, atol=TOL, rtol=TOL), (
        f"前向不等价: max |diff| = {max_diff:.3e} > {TOL}"
    )
    return max_diff


def main():
    parser = argparse.ArgumentParser(description="07-compose: TP x DP equivalence verification")
    parser.add_argument("--tp-size", type=int, default=2)
    parser.add_argument("--dp-size", type=int, default=None)
    args = parser.parse_args()

    # 非 torchrun 单进程直跑时补默认 rendezvous 环境变量（world_size=1）
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29515")
    rank, world_size, local_rank = setup()

    dp_size = args.dp_size if args.dp_size is not None else world_size // args.tp_size
    assert args.tp_size * dp_size == world_size, (
        f"tp_size({args.tp_size}) x dp_size({dp_size}) != world_size({world_size})；"
        f"可用 --tp-size/--dp-size 指定（4 进程默认 2x2，2 进程可用 1x2 或 2x1）"
    )
    tp_group, dp_group, tp_rank, dp_rank, _, _ = make_parallel_groups(world_size, args.tp_size)
    rank0_print(
        f"[verify] world_size={world_size} = dp({dp_size}) x tp({args.tp_size}), backend=gloo/CPU"
    )

    model_args = ModelArgs(
        n_layers=2, vocab_size=8, max_seq_len=16, dim=16, n_heads=4, dropout_p=0
    )
    # 固定数据：所有 rank 生成相同全量数据，各自只消费 dp_rank 分片
    batches = shard_batches(model_args, BATCH_SIZE, dp_size, num_batches=STEPS, seed=SEED)

    # 组合模型（前向检查与训练共用同一初始权重）
    _full, tp = build_models(model_args, SEED, tp_group)
    del _full

    # 1. 前向等价
    fwd_diff = check_forward_equivalence(model_args, batches, tp, dp_rank)
    rank0_print(f"[verify] forward equivalence: max |logits diff| = {fwd_diff:.3e} (< {TOL})")

    # 2. 训练等价：单进程参考 vs 组合并行
    ref_losses, ref_model = reference_training(model_args, batches, dp_size)
    tp_losses, tp_model_trained = composed_training(
        model_args, batches, tp_group, dp_group, dp_rank, dp_size
    )
    max_loss_diff = 0.0
    for step in range(STEPS):
        diff = abs(ref_losses[step] - tp_losses[step])
        max_loss_diff = max(max_loss_diff, diff)
        rank0_print(
            f"[verify] step {step}: ref loss {ref_losses[step]:.8f} | "
            f"tp+dp loss {tp_losses[step]:.8f} | diff {diff:.3e}"
        )
        assert diff < TOL, (
            f"step {step} loss 不等价: ref {ref_losses[step]:.8f} vs "
            f"tp+dp {tp_losses[step]:.8f}, diff {diff:.3e} > {TOL}"
        )
    rank0_print(f"[verify] training equivalence: max |loss diff| = {max_loss_diff:.3e} (< {TOL})")

    # 3. 训练后复制参数对比（DP 梯度同步正确性的直接证据：
    #    embedding/输出头若没在 DP 组内正确平均，多步 Adam 后会明显发散）
    for name in ("tok_embeddings.weight", "output.weight"):
        ref_w = dict(ref_model.named_parameters())[name]
        tp_w = dict(tp_model_trained.named_parameters())[name]
        wdiff = (ref_w - tp_w).abs().max().item()
        assert wdiff < WEIGHT_TOL, (
            f"训练后 {name} 不等价: max |diff| = {wdiff:.3e} > {WEIGHT_TOL}"
        )
        rank0_print(f"[verify] post-train {name}: max |diff| = {wdiff:.3e} (< {WEIGHT_TOL})")

    rank0_print(f"[verify] PASS (world_size={world_size}, dp={dp_size} x tp={args.tp_size})")
    cleanup()


if __name__ == "__main__":
    main()
