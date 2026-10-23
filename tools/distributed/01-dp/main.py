# -*- coding: utf-8 -*-

# ***************************************************
# * File        : main.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-08-20
# * Version     : 1.0.082007
# * Description : 01 单元：nn.DataParallel（DP）单机多卡数据并行。单进程内
# *               把 batch 切分到多块 GPU 上前向/反向，梯度在主卡聚合。
# *               DP 只支持单机多卡 CUDA；无 GPU 或单卡时入口直接报错退出。
# * Link        : https://docs.pytorch.org/tutorials/beginner/blitz/data_parallel_tutorial.html
# * Requirement : torch >= 2.5；CUDA GPU >= 2
# ***************************************************

__all__ = []

# python libraries
import sys
from pathlib import Path

# repo root: tools/distributed/01-dp/main.py -> parents[3]
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F

from tools.distributed._common.dataset import make_toy_dataloaders
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


def train(device: torch.device, num_epochs: int, batch_size: int) -> nn.Module:
    train_loader, test_loader = make_toy_dataloaders(batch_size=batch_size, distributed=False)
    model = ToyMLP(num_features=20, num_classes=2)

    # DP 的核心就这一行：batch 维度自动切分到各卡，梯度自动聚合回主卡，
    # 原模型可通过 model.module 访问
    model = nn.DataParallel(model)
    model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

    step = 0
    for epoch in range(num_epochs):
        model.train()
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(features)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            log_step_metrics(step=step, loss=loss.item(), extra=f"epoch {epoch}")
            step += 1

    train_acc = compute_accuracy(model, train_loader, device)
    test_acc = compute_accuracy(model, test_loader, device)
    print(f"Training accuracy: {train_acc:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")
    return model




# 测试代码 main 函数
def main():
    import argparse
    parser = argparse.ArgumentParser(description="01-dp: nn.DataParallel 单机多卡示例")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    # DP 只支持单机多卡 CUDA：无卡/单卡直接给出清晰报错
    if not torch.cuda.is_available():
        raise SystemExit(
            "01-dp 需要 CUDA GPU：nn.DataParallel 依赖 GPU 间通信，"
            "CPU/MPS 上无法运行。请在多卡机器上执行本脚本，"
            "或改用 02-ddp（支持 CPU + gloo 冒烟）。"
        )
    if torch.cuda.device_count() < 2:
        raise SystemExit(
            f"01-dp 需要至少 2 块 GPU（当前 {torch.cuda.device_count()} 块）："
            "单卡下 DataParallel 没有并行意义，请改用 00-single-gpu。"
        )

    device = torch.device("cuda:0")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Number of GPUs available: {torch.cuda.device_count()}")
    torch.manual_seed(123)

    train(device, args.epochs, args.batch_size)

if __name__ == "__main__":
    main()
