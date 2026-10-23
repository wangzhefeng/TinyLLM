# -*- coding: utf-8 -*-

# ***************************************************
# * File        : main.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-08-20
# * Version     : 1.0.082006
# * Description : 00 单元：单卡训练基线。用 _common 的 toy 数据集与 ToyMLP
# *               走一遍最普通的训练循环，作为后续 DP/DDP/FSDP 单元逐层
# *               加分布式能力的对照基线。单进程脚本，允许使用 MPS。
# * Link        : https://github.com/pytorch/examples/blob/main/distributed/ddp-tutorial-series/single_gpu.py
# * Requirement : torch >= 2.5
# ***************************************************

__all__ = []

# python libraries
import sys
from pathlib import Path

# repo root: tools/distributed/00-single-gpu/main.py -> parents[3]
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import torch
import torch.nn.functional as F

from tools.distributed._common.dataset import make_toy_dataloaders
from tools.distributed._common.device import get_device
from tools.distributed._common.metrics import log_step_metrics


def train(device: torch.device, num_epochs: int, batch_size: int, save_every: int) -> None:
    # 单卡基线：数据、模型、优化器都与分布式单元保持一致，便于对比
    train_loader, _ = make_toy_dataloaders(batch_size=batch_size, distributed=False)
    model = torch.nn.Linear(20, 2)  # 与 DDP 系列单元相同的最小模型
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-1)

    checkpoint_dir = None
    step = 0
    for epoch in range(num_epochs):
        model.train()
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            # forward / backward / step —— 单卡训练的全部内容
            optimizer.zero_grad()
            logits = model(features)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            log_step_metrics(step=step, loss=loss.item(), extra=f"epoch {epoch}")
            step += 1
        # 教学演示：周期性地保存 checkpoint
        if save_every > 0 and epoch % save_every == 0:
            from tools.distributed._common.dist import get_checkpoint_dir
            checkpoint_dir = get_checkpoint_dir("single-gpu")
            path = checkpoint_dir / "checkpoint.pt"
            torch.save(model.state_dict(), path)
            print(f"Epoch {epoch} | checkpoint saved at {path}")


# 测试代码 main 函数
def main():
    import argparse
    parser = argparse.ArgumentParser(description="00-single-gpu: 单卡训练基线")
    parser.add_argument("--epochs", type=int, default=2, help="训练轮数（默认 2，几秒内跑完）")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--save-every", type=int, default=0, help="每隔多少 epoch 保存一次 checkpoint，0 表示不保存")
    args = parser.parse_args()

    device = get_device(allow_mps=True)  # 单进程脚本允许 MPS；分布式单元禁止
    print(f"PyTorch version: {torch.__version__}")
    print(f"Use device: {device}")
    torch.manual_seed(123)

    train(device, args.epochs, args.batch_size, args.save_every)

if __name__ == "__main__":
    main()
