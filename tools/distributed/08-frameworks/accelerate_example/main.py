# -*- coding: utf-8 -*-

# ***************************************************
# * File        : main.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2026-10-24
# * Version     : 1.0.102401
# * Description : 08 单元：HuggingFace Accelerate 最小对照示例。
# *               与 02-ddp 手写实现对照：进程组初始化、模型 DDP 包装、
# *               数据切分、梯度同步、混合精度全部由 Accelerator 接管，
# *               训练循环本身与单卡代码几乎一致。
# *               用 run.sh（accelerate launch，2 进程 CPU + gloo）启动。
# * Link        : https://huggingface.co/docs/accelerate/basic_tutorials/install
# * Requirement : torch >= 2.5；accelerate >= 1.5（已装入 .venv）
# ***************************************************

__all__ = []

# python libraries
import sys
from pathlib import Path

# repo root: tools/distributed/08-frameworks/accelerate_example/main.py -> parents[4]
ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import torch
import torch.nn.functional as F
from accelerate import Accelerator

from tools.distributed._common.dataset import make_toy_dataloaders
from tools.distributed._common.model import ToyMLP


def evaluate(accelerator, model, test_loader) -> float:
    """各 rank 持有完整测试集，独立评估精度（梯度已同步，结果一致）。"""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for features, labels in test_loader:
            features = features.to(accelerator.device)
            labels = labels.to(accelerator.device)
            preds = model(features).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.numel()
    return correct / total


# 测试代码 main 函数
def main():
    import argparse
    parser = argparse.ArgumentParser(description="08-frameworks: Accelerate 最小示例")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-1)
    args = parser.parse_args()

    # 对照 02-ddp 的 dist_utils.setup()：
    # Accelerator 按启动环境（accelerate launch 注入的变量）自动完成
    # 进程组初始化（CPU 走 gloo）与设备绑定，无需手写 RANK/WORLD_SIZE 处理。
    accelerator = Accelerator()
    accelerator.print(f"world_size={accelerator.num_processes}, device={accelerator.device}")

    # 注意：这里不要自己套 DistributedSampler；prepare() 会把 dataloader
    # 包装成分布式分片版本（等价于 02-ddp 的 distributed=True + set_epoch）。
    train_loader, test_loader = make_toy_dataloaders(batch_size=args.batch_size)

    model = ToyMLP()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

    # 对照 02-ddp 的 DDP(model)：prepare 统一完成模型分布式包装、
    # dataloader 分片、优化器/混合精度适配（本示例 mixed_precision=no）。
    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)

    step = 0
    for epoch in range(args.epochs):
        model.train()
        for features, labels in train_loader:
            # prepare 后的 batch 已在正确设备上，无需手动 .to(device)
            optimizer.zero_grad()
            logits = model(features)
            loss = F.cross_entropy(logits, labels)
            # 对照 02-ddp 的 loss.backward()：框架负责梯度同步
            # （以及混合精度下的 loss 缩放，本示例未开启）。
            accelerator.backward(loss)
            optimizer.step()
            if accelerator.is_main_process and step % 10 == 0:
                print(f"step {step:>5d} | loss {loss.item():.4f} | epoch {epoch}")
            step += 1
        acc = evaluate(accelerator, model, test_loader)
        accelerator.print(f"epoch {epoch} | test acc {acc:.4f}")

    accelerator.wait_for_everyone()
    # 进程组随进程退出自动清理，无需手写 destroy_process_group。

if __name__ == "__main__":
    main()
