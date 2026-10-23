# -*- coding: utf-8 -*-

# ***************************************************
# * File        : verify_equivalence.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2026-10-24
# * Version     : 1.0.102404
# * Description : 05 单元：张量并行的数值等价验证（硬性自检脚本）。
# *               固定 seed 在每个 rank 上构造逐位相同的完整模型，并
# *               从中切出本 rank 的 TP 分片（数学上同一个模型）：
# *                 1. 前向等价：同一输入下 TP 模型与完整模型的 logits
# *                    必须在 1e-5 容差内一致（每个 rank 各自断言）；
# *                    每个 rank 独立跑完整模型，即等价于单进程参考。
# *                 2. 训练等价：同一批固定数据、同样的优化器各训练
# *                    若干步，loss 序列必须逐步一致。
# *                 3. 不整除检查：n_heads 不能被 world_size 整除时
# *                    必须抛出断言错误。
# *               torchrun 2 进程（CPU+gloo）与单进程均可运行：
# *                 .venv/bin/python -m torch.distributed.run \
# *                     --nproc_per_node=2 --master_port=29513 \
# *                     tools/distributed/05-tp/verify_equivalence.py
# * Link        : https://arxiv.org/abs/1909.08053 (Megatron-LM)
# * Requirement : torch >= 2.5；gloo（CPU）或 nccl（CUDA）后端
# ***************************************************

# python libraries
import os
import sys
from pathlib import Path

# repo root: tools/distributed/05-tp/verify_equivalence.py -> parents[3]
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import torch
import torch.nn.functional as F

from parallel_linear import ColumnParallelLinear
from tp_model import TPAttention, TPTransformer
from tools.distributed._common.dist import cleanup, rank0_print, setup
from tools.distributed._common.model import ModelArgs, Transformer

SEED = 1234
STEPS = 5
TOL = 1e-5


def build_models() -> tuple[Transformer, TPTransformer, ModelArgs]:
    """固定 seed：每个 rank 上的完整模型逐位相同，TP 分片自它切出。"""
    model_args = ModelArgs(
        n_layers=2, vocab_size=8, max_seq_len=16, dim=16, n_heads=4, dropout_p=0
    )
    torch.manual_seed(SEED)
    full = Transformer(model_args)
    tp = TPTransformer(model_args)
    tp.load_from_full(full)
    return full, tp, model_args


def fixed_data(model_args: ModelArgs, num_batches: int = 2):
    """固定输入/目标（seed 确定，所有 rank 完全一致）。"""
    gen = torch.Generator().manual_seed(SEED + 1)
    data = []
    for _ in range(num_batches):
        tokens = torch.randint(0, model_args.vocab_size, (4, 16), generator=gen)
        targets = torch.randint(0, model_args.vocab_size, (4, 16), generator=gen)
        data.append((tokens, targets))
    return data


def check_forward_equivalence(full, tp, model_args, world_size):
    """第 1 部分：前向 logits 等价（eval 模式，消除 dropout 影响）。"""
    full.eval()
    tp.eval()
    tokens, _ = fixed_data(model_args, 1)[0]
    with torch.no_grad():
        ref_logits = full(tokens)  # 每个 rank 独立计算，等价于单进程参考
        tp_logits = tp(tokens)
    assert ref_logits.shape == tp_logits.shape, (
        f"形状不一致: full {ref_logits.shape} vs tp {tp_logits.shape}"
    )
    max_diff = (ref_logits - tp_logits).abs().max().item()
    assert torch.allclose(tp_logits, ref_logits, atol=TOL, rtol=TOL), (
        f"前向不等价: max |diff| = {max_diff:.3e} > {TOL}"
    )
    return max_diff


def check_training_equivalence(full, tp, model_args, world_size):
    """第 2 部分：loss 序列逐步等价。"""
    # 重新构造，保证两部分互不影响（同样的 seed，同样的初始权重）
    full, tp, model_args = build_models()
    full.train()
    tp.train()
    data = fixed_data(model_args, num_batches=STEPS)
    full_optim = torch.optim.Adam(full.parameters(), lr=1e-2)
    tp_optim = torch.optim.Adam(tp.parameters(), lr=1e-2)

    max_loss_diff = 0.0
    losses = []
    for step in range(STEPS):
        tokens, targets = data[step]

        ref_logits = full(tokens)
        ref_loss = F.cross_entropy(
            ref_logits.view(-1, model_args.vocab_size), targets.view(-1)
        )
        ref_loss.backward()
        full_optim.step()
        full_optim.zero_grad()

        tp_logits = tp(tokens)
        tp_loss = F.cross_entropy(
            tp_logits.view(-1, model_args.vocab_size), targets.view(-1)
        )
        tp_loss.backward()
        tp_optim.step()
        tp_optim.zero_grad()

        diff = abs(ref_loss.item() - tp_loss.item())
        max_loss_diff = max(max_loss_diff, diff)
        losses.append((step, ref_loss.item(), tp_loss.item(), diff))
        assert diff < TOL, (
            f"step {step} loss 不等价: full {ref_loss.item():.8f} vs "
            f"tp {tp_loss.item():.8f}, diff {diff:.3e} > {TOL}"
        )
    return losses, max_loss_diff


def check_non_divisible_assertion(world_size):
    """第 3 部分：不整除时必须断言报错（仅在 world_size>1 时有意义）。"""
    if world_size == 1:
        return
    bad_args = ModelArgs(dim=16, n_heads=3)  # 3 个头无法均分到 2 个 rank
    try:
        TPAttention(bad_args)
    except AssertionError:
        return
    raise RuntimeError("n_heads=3 / world_size=2 未触发断言，整除检查失效")
    # ColumnParallelLinear 的断言同理（out_features 不整除）
    try:
        ColumnParallelLinear(8, 6)
    except AssertionError:
        return
    raise RuntimeError("out_features=6 / world_size=2 未触发断言，整除检查失效")


def main():
    # 非 torchrun 单进程直跑时补默认 rendezvous 环境变量（world_size=1）
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29513")
    rank, world_size, local_rank = setup()
    rank0_print(f"[verify] world_size={world_size}, backend=gloo/CPU")

    full, tp, model_args = build_models()

    # 1. 前向等价
    fwd_diff = check_forward_equivalence(full, tp, model_args, world_size)
    rank0_print(f"[verify] forward equivalence: max |logits diff| = {fwd_diff:.3e} (< {TOL})")

    # 2. 训练等价
    losses, max_loss_diff = check_training_equivalence(full, tp, model_args, world_size)
    for step, ref_loss, tp_loss, diff in losses:
        rank0_print(
            f"[verify] step {step}: full loss {ref_loss:.8f} | "
            f"tp loss {tp_loss:.8f} | diff {diff:.3e}"
        )
    rank0_print(f"[verify] training equivalence: max |loss diff| = {max_loss_diff:.3e} (< {TOL})")

    # 3. 不整除断言
    check_non_divisible_assertion(world_size)
    rank0_print("[verify] non-divisible assertion check OK")

    rank0_print(f"[verify] PASS (world_size={world_size})")
    cleanup()


if __name__ == "__main__":
    main()
