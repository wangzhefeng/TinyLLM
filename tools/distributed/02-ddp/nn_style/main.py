# -*- coding: utf-8 -*-

# ***************************************************
# * File        : main.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-08-20
# * Version     : 1.0.082011
# * Description : 02 单元（作者模块化版）：DDP 单机多卡，mp.spawn 启动。
# *               进程组初始化复用仓库主链路 utils/ddp_utils.py，数据集与
# *               模型来自 tools/distributed/_common/；训练结束后各 rank
# *               独立计算训练/测试准确率。
# * Link        : https://github.com/rasbt/LLMs-from-scratch/blob/main/appendix-A/03_main-chapter-code/DDP-script.py
# * Requirement : torch >= 2.5；gloo（CPU）或 nccl（CUDA）
# ***************************************************

__all__ = []

# python libraries
import sys
from pathlib import Path

# repo root: tools/distributed/02-ddp/nn_style/main.py -> parents[4]
ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

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
        # 每个 epoch 设置 sampler 种子，保证各 rank 采样一致且随 epoch 变化
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

    train_acc = compute_accuracy(model, train_loader, device)
    test_acc = compute_accuracy(model, test_loader, device)
    print(f"[rank {rank}] Training accuracy: {train_acc:.4f}")
    print(f"[rank {rank}] Test accuracy: {test_acc:.4f}")

    destory_dist()  # 注意：函数名拼写沿用 utils/ddp_utils.py 现状（另案处理）


# 测试代码 main 函数
def main():
    import argparse
    parser = argparse.ArgumentParser(description="02-ddp/nn_style: DDP mp.spawn 版")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--nprocs", type=int, default=0,
                        help="进程数；默认 0 表示 CUDA 可用时取 GPU 数，否则取 2（CPU 冒烟）")
    args = parser.parse_args()

    torch.manual_seed(123)
    world_size = args.nprocs or (torch.cuda.device_count() if torch.cuda.is_available() else 2)
    if world_size < 2:
        raise SystemExit("DDP 至少需要 2 个进程；单进程请改用 00-single-gpu。")
    mp.spawn(train, args=(world_size, args.epochs, args.batch_size), nprocs=world_size)

if __name__ == "__main__":
    main()
