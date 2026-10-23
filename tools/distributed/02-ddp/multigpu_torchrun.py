# -*- coding: utf-8 -*-

# ***************************************************
# * File        : multigpu_torchrun.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-08-20
# * Version     : 1.0.082009
# * Description : 02 单元：DDP 单机多进程（torchrun 启动，推荐方式）。从环境
# *               变量读取 RANK/WORLD_SIZE/LOCAL_RANK；带 snapshot 断点续训
# *               演示，snapshot 保存在仓库根的
# *               results/checkpoints/distributed/ddp/（与 cwd 无关）。
# *               无 GPU 时自动回退 CPU + gloo。
# * Link        : https://github.com/pytorch/examples/blob/main/distributed/ddp-tutorial-series/multigpu_torchrun.py
# * Requirement : torch >= 2.5；gloo（CPU）或 nccl（CUDA）
# ***************************************************

__all__ = []

# python libraries
import os
import sys
from pathlib import Path

# repo root: tools/distributed/02-ddp/multigpu_torchrun.py -> parents[3]
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from tools.distributed._common import dist as dist_utils
from tools.distributed._common.dataset import make_toy_dataloaders
from tools.distributed._common.device import get_dist_device
from tools.distributed._common.metrics import log_step_metrics


class Trainer:
    """
    与 00 单卡基线相同的训练循环，外加：
    - snapshot 保存/恢复（MODEL_STATE + EPOCHS_RUN），仅 rank 0 写盘
    """

    def __init__(self, model, train_loader, optimizer, device, local_rank, snapshot_path):
        self.device = device
        self.local_rank = local_rank
        self.model = model.to(device)
        self.train_loader = train_loader
        self.optimizer = optimizer
        self.epochs_run = 0
        self.snapshot_path = snapshot_path
        if os.path.exists(snapshot_path):
            dist_utils.rank0_print("Loading snapshot")
            self._load_snapshot()
        use_cuda = device.type == "cuda"
        self.model = DDP(self.model, device_ids=[local_rank] if use_cuda else None)

    def _load_snapshot(self):
        snapshot = torch.load(self.snapshot_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        dist_utils.rank0_print(f"Resuming training from snapshot at epoch {self.epochs_run}")

    def _save_snapshot(self, epoch):
        snapshot = {
            "MODEL_STATE": self.model.module.state_dict(),
            "EPOCHS_RUN": epoch + 1,  # 下次从下一个 epoch 继续
        }
        torch.save(snapshot, self.snapshot_path)
        dist_utils.rank0_print(f"Epoch {epoch} | snapshot saved at {self.snapshot_path}")

    def train(self, max_epochs: int, save_every: int):
        step = 0
        for epoch in range(self.epochs_run, max_epochs):
            self.train_loader.sampler.set_epoch(epoch)
            self.model.train()
            for features, labels in self.train_loader:
                features = features.to(self.device)
                labels = labels.to(self.device)
                self.optimizer.zero_grad()
                logits = self.model(features)
                loss = F.cross_entropy(logits, labels)
                loss.backward()
                self.optimizer.step()
                log_step_metrics(step=step, loss=loss.item(), extra=f"epoch {epoch}")
                step += 1
            if dist_utils.is_main_process() and save_every > 0 and (epoch + 1) % save_every == 0:
                self._save_snapshot(epoch)


# 测试代码 main 函数
def main():
    import argparse
    parser = argparse.ArgumentParser(description="02-ddp: DDP 单机多进程（torchrun）示例")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--save-every", type=int, default=1, help="每隔多少 epoch 保存 snapshot，0 表示不保存")
    args = parser.parse_args()

    rank, world_size, local_rank = dist_utils.setup()
    device = get_dist_device(local_rank)

    snapshot_path = str(dist_utils.get_checkpoint_dir("ddp") / "snapshot.pt")
    train_loader, _ = make_toy_dataloaders(batch_size=args.batch_size, distributed=True)
    model = torch.nn.Linear(20, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-1)

    trainer = Trainer(model, train_loader, optimizer, device, local_rank, snapshot_path)
    trainer.train(args.epochs, args.save_every)
    dist_utils.cleanup()

if __name__ == "__main__":
    main()
