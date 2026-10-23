# -*- coding: utf-8 -*-

# ***************************************************
# * File        : train.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2026-10-24
# * Version     : 1.0.102401
# * Description : 03 单元统一入口：--stage {0,1,2,3} 选择 ZeRO 实现，
# *               四个 stage 共用同一份训练循环骨架（模型、数据、日志、
# *               显存账、等价性比对），stage 之间的差别全部收敛在
# *               zero1/zero2/zero3 的 Trainer 里。
# *               torchrun 启动，CPU + gloo 即可运行：
# *                 python -m torch.distributed.run --nproc_per_node=2 \
# *                     --master_port=29512 tools/distributed/03-zero/train.py \
# *                     --stage 1 --epochs 1
# * Link        : https://arxiv.org/abs/1910.02054
# * Requirement : torch >= 2.5；gloo（CPU）或 nccl（CUDA）
# ***************************************************

__all__ = []

# python libraries
import sys
from pathlib import Path

# repo root & 本目录（03-zero 目录名含数字与短横线，不能作为包导入）
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for p in (str(ROOT), str(HERE)):
    if p not in sys.path:
        sys.path.append(p)

import torch

from tools.distributed._common import dist as dist_utils
from tools.distributed._common.dataset import make_toy_dataloaders
from tools.distributed._common.metrics import log_step_metrics
from tools.distributed._common.model import ToyMLP

import zero_common
import zero1
import zero2
import zero3

STAGES = {
    0: ("stage 0: DDP 基线（参数/梯度/优化器状态全量复制）", zero_common.FullReplicaTrainer),
    1: ("stage 1: ZeRO-1 优化器状态分片", zero1.ZeroOneTrainer),
    2: ("stage 2: ZeRO-2 + 梯度分片", zero2.ZeroTwoTrainer),
    3: ("stage 3: ZeRO-3 + 参数分片", zero3.ZeroThreeTrainer),
}


# 测试代码 main 函数
def main():
    import argparse
    parser = argparse.ArgumentParser(description="03-zero: ZeRO stage 0/1/2/3 手写教学实现")
    parser.add_argument("--stage", type=int, required=True, choices=[0, 1, 2, 3])
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--steps", type=int, default=0, help="最多训练多少 step，0 表示不限（按 epochs 跑完）")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verify-ref", type=str, default="",
                        help="单进程基线参考文件（verify_equivalence.py 生成）；非空则训练后比对并给出 PASS/FAIL")
    args = parser.parse_args()

    rank, world_size, _local_rank = dist_utils.setup()
    stage_name, trainer_cls = STAGES[args.stage]
    dist_utils.rank0_print(f"[03-zero] {stage_name} | world_size={world_size} lr={args.lr} seed={args.seed}")

    # 数据说明：所有 rank 使用全量数据（distributed=False）+ 相同 seed，
    # 梯度靠集合通信手工平均，数学上与单进程整批训练严格等价。
    # 这是为了验证分片实现的正确性，而不是数据并行效率——效率做法是
    # 02-ddp 的 DistributedSampler 切分数据。
    torch.manual_seed(args.seed)
    model = ToyMLP()
    train_loader, _ = make_toy_dataloaders(batch_size=args.batch_size, distributed=False, seed=args.seed)
    trainer = trainer_cls(model, rank, world_size, lr=args.lr)

    step = 0
    last_loss = None
    stop = False
    for epoch in range(args.epochs):
        model.train()
        for features, labels in train_loader:
            last_loss = trainer.train_step(features, labels)
            log_step_metrics(step=step, loss=last_loss, extra=f"stage {args.stage} epoch {epoch}")
            step += 1
            if args.steps and step >= args.steps:
                stop = True
                break
        if stop:
            break

    zero_common.print_memory_report(args.stage, world_size, trainer.memory_report())

    exit_code = 0
    if args.verify_ref:
        flat = trainer.final_flat_params()
        ok, msg = zero_common.verify_against_ref(flat, last_loss, args.verify_ref)
        if dist_utils.is_main_process():
            print(f"[verify] stage {args.stage} vs 单进程基线: {msg} -> {'PASS' if ok else 'FAIL'}")
        exit_code = 0 if ok else 1

    dist_utils.cleanup()
    sys.exit(exit_code)

if __name__ == "__main__":
    main()
