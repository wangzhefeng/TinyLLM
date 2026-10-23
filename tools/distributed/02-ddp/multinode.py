# -*- coding: utf-8 -*-

# ***************************************************
# * File        : multinode.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-08-20
# * Version     : 1.0.082010
# * Description : 02 单元：DDP 多机多卡（torchrun 每节点各启一份）。与
# *               multigpu_torchrun.py 的唯一区别是显式区分 global rank 与
# *               local rank：设备绑定用 LOCAL_RANK，日志/存档判断用全局
# *               RANK。启动命令见文件底部注释。
# * Link        : https://github.com/pytorch/examples/blob/main/distributed/ddp-tutorial-series/multinode.py
# * Requirement : torch >= 2.5；gloo（CPU）或 nccl（CUDA）
# ***************************************************

# 多机启动示例（两个节点，每节点 2 进程）：
#   节点 0（master）：
#     torchrun --nnodes=2 --nproc_per_node=2 --rdzv_id=100 \
#         --rdzv_backend=c10d --rdzv_endpoint=<node0-ip>:29400 \
#         tools/distributed/02-ddp/multinode.py
#   节点 1：同上命令（rdzv_endpoint 指向节点 0）

__all__ = []

# python libraries
import os
import sys
from pathlib import Path

# repo root: tools/distributed/02-ddp/multinode.py -> parents[3]
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
    多机场景：
    - local_rank  -> 绑定本机设备（cuda:{local_rank}）、DDP device_ids
    - global_rank -> 决定谁打日志、谁存 snapshot（全局 rank 0）
    """

    def __init__(self, model, train_loader, optimizer, device, local_rank, snapshot_path):
        self.device = device
        self.local_rank = local_rank
        self.global_rank = dist_utils.get_rank()
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
            "EPOCHS_RUN": epoch + 1,
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
                log_step_metrics(step=step, loss=loss.item(),
                                 extra=f"epoch {epoch} | node-rank {self.global_rank}")
                step += 1
            if dist_utils.is_main_process() and save_every > 0 and (epoch + 1) % save_every == 0:
                self._save_snapshot(epoch)


# 测试代码 main 函数
def main():
    import argparse
    parser = argparse.ArgumentParser(description="02-ddp: DDP 多机多卡示例")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--save-every", type=int, default=1)
    args = parser.parse_args()

    rank, world_size, local_rank = dist_utils.setup()
    device = get_dist_device(local_rank)
    dist_utils.rank0_print(f"world size: {world_size} (multi-node)")

    snapshot_path = str(dist_utils.get_checkpoint_dir("ddp") / "snapshot_multinode.pt")
    train_loader, _ = make_toy_dataloaders(batch_size=args.batch_size, distributed=True)
    model = torch.nn.Linear(20, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-1)

    trainer = Trainer(model, train_loader, optimizer, device, local_rank, snapshot_path)
    trainer.train(args.epochs, args.save_every)
    dist_utils.cleanup()

if __name__ == "__main__":
    main()
