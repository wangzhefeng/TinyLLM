# -*- coding: utf-8 -*-

# ***************************************************
# * File        : multigpu_spawn.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-08-20
# * Version     : 1.0.082008
# * Description : 02 单元：DDP 单机多卡（mp.spawn 启动）。torch.multiprocessing
# *               .spawn 为每个进程 fork 一个解释器并把 rank 传入子进程；
# *               与 00 单卡基线的差别仅在于：进程组初始化、
# *               DistributedSampler 切数据、DDP 包模型（已在注释中标出）。
# *               无 GPU 时自动回退 CPU + gloo，便于本地冒烟。
# * Link        : https://github.com/pytorch/examples/blob/main/distributed/ddp-tutorial-series/multigpu.py
# * Requirement : torch >= 2.5；gloo（CPU）或 nccl（CUDA）
# ***************************************************

__all__ = []

# python libraries
import sys
from pathlib import Path

# repo root: tools/distributed/02-ddp/multigpu_spawn.py -> parents[3]
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from tools.distributed._common import dist as dist_utils
from tools.distributed._common.dataset import make_toy_dataloaders
from tools.distributed._common.device import get_dist_device
from tools.distributed._common.metrics import log_step_metrics


def train(rank: int, world_size: int, num_epochs: int, batch_size: int) -> None:
    # 【DDP-1】初始化进程组：spawn 场景下 rank 由 mp.spawn 传入
    dist_utils.setup_spawn(rank, world_size)
    device = get_dist_device(rank)
    use_cuda = device.type == "cuda"

    # 【DDP-2】数据：DistributedSampler 把样本不重叠地切给各 rank
    train_loader, _ = make_toy_dataloaders(batch_size=batch_size, distributed=True)
    model = torch.nn.Linear(20, 2)
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-1)

    # 【DDP-3】模型：DDP 包装后，反向传播时自动跨进程 all-reduce 梯度
    model = DDP(model, device_ids=[rank] if use_cuda else None)

    step = 0
    for epoch in range(num_epochs):
        # 每个 epoch 重新打乱，保证各 rank 的采样顺序一致且随 epoch 变化
        train_loader.sampler.set_epoch(epoch)
        model.train()
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(features)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            if rank == 0:
                log_step_metrics(step=step, loss=loss.item(), extra=f"epoch {epoch}")
            step += 1

    dist_utils.cleanup()


# 测试代码 main 函数
def main():
    import argparse
    parser = argparse.ArgumentParser(description="02-ddp: DDP 单机多进程（mp.spawn）示例")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--nprocs", type=int, default=0,
                        help="进程数；默认 0 表示 CUDA 可用时取 GPU 数，否则取 2（CPU 冒烟）")
    args = parser.parse_args()

    world_size = args.nprocs or (torch.cuda.device_count() if torch.cuda.is_available() else 2)
    if world_size < 2:
        raise SystemExit("DDP 至少需要 2 个进程；单进程请改用 00-single-gpu。")
    dist_utils.rank0_print(f"spawning {world_size} processes (backend auto: nccl/gloo)")
    mp.spawn(train, args=(world_size, args.epochs, args.batch_size), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()
