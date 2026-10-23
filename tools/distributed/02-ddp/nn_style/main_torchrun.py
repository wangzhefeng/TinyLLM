# -*- coding: utf-8 -*-

# ***************************************************
# * File        : main_torchrun.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-08-20
# * Version     : 1.0.082012
# * Description : 02 单元（作者模块化版）：DDP 单机多进程，torchrun 启动。
# *               进程组初始化复用 utils/ddp_utils.py；从 torchrun 设置的
# *               环境变量读取 LOCAL_RANK。仅 rank 0 打印训练日志。
# * Link        : https://github.com/rasbt/LLMs-from-scratch/blob/main/appendix-A/03_main-chapter-code/DDP-script-torchrun.py
# * Requirement : torch >= 2.5；gloo（CPU）或 nccl（CUDA）
# ***************************************************

__all__ = []

# python libraries
import os
import sys
from pathlib import Path

# repo root: tools/distributed/02-ddp/nn_style/main_torchrun.py -> parents[4]
ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.elastic.multiprocessing.errors import record

from utils.ddp_utils import ddp_setup_custom, destory_dist
from tools.distributed._common.dataset import make_toy_dataloaders
from tools.distributed._common.device import get_dist_device
from tools.distributed._common.metrics import log_step_metrics
from tools.distributed._common.model import ToyMLP


def compute_accuracy(model, dataloader, device):
    model.eval()
    correct = 0.0
    total = 0
    for features, labels in dataloader:
        features, labels = features.to(device), labels.to(device)
        with torch.no_grad():
            logits = model(features)
        correct += torch.sum(torch.argmax(logits, dim=1) == labels)
        total += len(labels)
    return (correct / total).item()


def train(rank, world_size, num_epochs, batch_size):
    # 进程组初始化：复用 utils/ddp_utils.py（自动选 nccl/gloo）
    ddp_setup_custom(rank, world_size)
    device = get_dist_device(rank)
    use_cuda = device.type == "cuda"

    train_loader, test_loader = make_toy_dataloaders(batch_size=batch_size, distributed=True)
    model = ToyMLP(num_features=20, num_classes=2)
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

    model = DDP(model, device_ids=[rank] if use_cuda else None)

    step = 0
    for epoch in range(num_epochs):
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

    # 每个 rank 独立在完整测试集上评估（结果应一致）
    train_acc = compute_accuracy(model, train_loader, device)
    test_acc = compute_accuracy(model, test_loader, device)
    print(f"[rank {rank}] Training accuracy: {train_acc:.4f}")
    print(f"[rank {rank}] Test accuracy: {test_acc:.4f}")

    destory_dist()  # 注意：函数名拼写沿用 utils/ddp_utils.py 现状（另案处理）


# 测试代码 main 函数
@record
def main():
    import argparse
    parser = argparse.ArgumentParser(description="02-ddp/nn_style: DDP torchrun 版")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    # torchrun 会设置这些环境变量；缺省回退单进程
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", 0)))
    if rank == 0:
        print(f"PyTorch version: {torch.__version__}")
        print(f"world size: {world_size}")
        print(f"CUDA available: {torch.cuda.is_available()}")

    torch.manual_seed(123)
    train(rank, world_size, args.epochs, args.batch_size)

if __name__ == "__main__":
    main()
