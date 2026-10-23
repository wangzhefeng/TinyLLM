# -*- coding: utf-8 -*-

# ***************************************************
# * File        : verify_equivalence.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2026-10-24
# * Version     : 1.0.102401
# * Description : 03 单元等价性验证：固定相同 seed，先在单进程内用同一
# *               手写 Adam 跑 N 步作为基线，再用 torchrun 2 进程分别跑
# *               stage 0/1/2/3，断言各 stage 的最终参数向量与最终 loss
# *               和单进程基线在容差内一致，全部通过则输出 PASS。
# *               等价性成立的前提：所有 rank 用全量数据 + 梯度手工平均
# *               （见 train.py 数据说明），数学上与单进程整批训练一致。
# *               直接运行：.venv/bin/python tools/distributed/03-zero/verify_equivalence.py
# * Link        : https://arxiv.org/abs/1910.02054
# * Requirement : torch >= 2.5；gloo（CPU）
# ***************************************************

__all__ = []

# python libraries
import subprocess
import sys
from pathlib import Path

# repo root & 本目录（03-zero 目录名含数字与短横线，不能作为包导入）
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for p in (str(ROOT), str(HERE)):
    if p not in sys.path:
        sys.path.append(p)

import torch
import torch.nn.functional as F

from tools.distributed._common.dataset import make_toy_dataloaders
from tools.distributed._common.model import ToyMLP

from zero_common import ShardedAdam


def run_single_process_baseline(steps: int, batch_size: int, lr: float, seed: int):
    """
    单进程基线：不初始化进程组，用与各 stage 完全相同的手写 Adam
    （ShardedAdam 作用于完整参数，等价于整向量 Adam）训练 steps 步。
    数据顺序、模型初始化由各 seed 保证与分布式运行一致。
    """
    torch.manual_seed(seed)
    model = ToyMLP()
    train_loader, _ = make_toy_dataloaders(batch_size=batch_size, distributed=False, seed=seed)
    params = list(model.parameters())
    opt = ShardedAdam(params, lr=lr)

    losses = []
    done = False
    while not done:
        for features, labels in train_loader:
            model.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(features), labels)
            loss.backward()
            opt.step([p.grad for p in params])
            losses.append(loss.item())
            if len(losses) >= steps:
                done = True
                break
    flat = torch.cat([p.detach().reshape(-1) for p in params])
    return flat, losses


def run_stage_distributed(stage: int, steps: int, batch_size: int, lr: float,
                          seed: int, nproc: int, ref_path: Path) -> tuple[bool, str]:
    """用 torchrun 跑一个 stage 的 --verify-ref 模式，返回 (是否通过, [verify] 行)。"""
    cmd = [
        sys.executable, "-m", "torch.distributed.run",
        f"--nproc_per_node={nproc}",
        f"--master_port={29710 + stage}",
        str(HERE / "train.py"),
        "--stage", str(stage),
        "--steps", str(steps),
        "--batch-size", str(batch_size),
        "--lr", str(lr),
        "--seed", str(seed),
        "--verify-ref", str(ref_path),
    ]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=600)
    verify_lines = [l for l in proc.stdout.splitlines() if "[verify]" in l]
    detail = verify_lines[-1] if verify_lines else proc.stderr.strip().splitlines()[-1][:200]
    return proc.returncode == 0, detail


# 测试代码 main 函数
def main():
    import argparse
    parser = argparse.ArgumentParser(description="03-zero: 单进程基线 vs 各 ZeRO stage 的等价性验证")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--nproc", type=int, default=2)
    parser.add_argument("--stages", type=str, default="0,1,2,3")
    args = parser.parse_args()

    print(f"[verify] 单进程基线：steps={args.steps} batch_size={args.batch_size} lr={args.lr} seed={args.seed}")
    flat, losses = run_single_process_baseline(args.steps, args.batch_size, args.lr, args.seed)
    print(f"[verify] 基线 loss: {losses[0]:.4f} (step 0) -> {losses[-1]:.4f} (step {len(losses) - 1})")

    ref_path = HERE / "verify_ref_baseline.pt"
    torch.save({"flat": flat, "losses": losses}, ref_path)

    results = {}
    for stage in [int(s) for s in args.stages.split(",")]:
        ok, detail = run_stage_distributed(
            stage, args.steps, args.batch_size, args.lr, args.seed, args.nproc, ref_path)
        results[stage] = ok
        print(f"  {detail}")

    ref_path.unlink(missing_ok=True)
    print("\n[verify] 汇总：")
    for stage, ok in results.items():
        print(f"  stage {stage}: {'PASS' if ok else 'FAIL'}")
    if all(results.values()):
        print("[verify] PASS：所有 stage 与单进程基线在容差内一致")
        sys.exit(0)
    print("[verify] FAIL：存在与基线不一致的 stage")
    sys.exit(1)

if __name__ == "__main__":
    main()
