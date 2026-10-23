# -*- coding: utf-8 -*-

# ***************************************************
# * File        : zero1.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2026-10-24
# * Version     : 1.0.102401
# * Description : 03 单元 ZeRO stage 1：优化器状态分片（P_os）。
# *               分片了什么：Adam 的 exp_avg/exp_avg_sq，每 rank 只持有
# *               1/w（w = world_size），并只更新自己负责的参数分片；
# *               没分片什么：参数本体与梯度仍是全量（各 rank 一份）。
# *               显存账（fp32 numel 计）：优化器状态 2N → 2N/w，
# *               参数 N、梯度 N 不变。通信量与 DDP 相同（梯度 all-reduce
# *               换成 reduce-scatter + 参数 all-gather，总量一致）。
# * Link        : https://arxiv.org/abs/1910.02054
# * Requirement : torch >= 2.5；gloo（CPU）或 nccl（CUDA）
# ***************************************************

__all__ = ["ZeroOneTrainer"]

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
import torch.distributed as dist
import torch.nn.functional as F

from zero_common import ShardedAdam, pad_to_multiple, all_gather_shards


class ZeroOneTrainer:
    """
    ZeRO stage 1：只分片优化器状态。

    每步流程（对照注释看通信发生在哪）：
      1. 前向/反向在完整参数上进行（各 rank 持全量参数副本）；
      2. 梯度拼平后 all-reduce 求平均——本 stage 仍保留全量梯度；
      3. 只取自己负责的分片段梯度，在分片上执行 ShardedAdam 更新；
      4. 各 rank 把更新后的参数分片 all-gather 拼回完整参数。

    拼平主副本：初始化时让所有 nn.Parameter 的 .data 成为同一段连续
    存储的视图，因此分片更新与 all-gather 直接作用于参数本体，
    无需在"平铺副本"和"模型参数"之间来回拷贝。
    """

    def __init__(self, model: torch.nn.Module, rank: int, world_size: int, lr: float):
        self.model = model
        self.rank = rank
        self.world_size = world_size
        self.params = list(model.parameters())
        self.total = sum(p.numel() for p in self.params)
        padded = pad_to_multiple(self.total, world_size)
        self.shard_size = padded // world_size
        self.start = rank * self.shard_size

        # 拼平主副本 + 参数视图重指（见类 docstring）
        self.flat = torch.zeros(padded)
        offset = 0
        with torch.no_grad():
            for p in self.params:
                n = p.numel()
                self.flat[offset:offset + n].copy_(p.detach().reshape(-1))
                offset += n
        offset = 0
        for p in self.params:
            n = p.numel()
            p.data = self.flat[offset:offset + n].view_as(p)
            offset += n

        # 本 rank 负责的参数分片（flat 的视图）；优化器状态只有 2×分片长
        self.shard = self.flat[self.start:self.start + self.shard_size]
        self.opt = ShardedAdam([self.shard], lr=lr)

    def _flatten_grads(self) -> torch.Tensor:
        """把各参数的梯度拼平并补零到等长分片的整数倍。"""
        grads = []
        for p in self.params:
            assert p.grad is not None, "存在未参与反向的参数，本教学实现不支持"
            grads.append(p.grad.detach().reshape(-1))
        flat_grad = torch.cat(grads)
        if flat_grad.numel() < self.flat.numel():
            flat_grad = torch.cat([flat_grad, torch.zeros(self.flat.numel() - flat_grad.numel())])
        return flat_grad

    def _sync_grads(self, flat_grad: torch.Tensor) -> torch.Tensor:
        """stage 1 的梯度同步：全量 all-reduce 求平均，然后只取自己的分片段。
        （各 rank 瞬时仍持有全量梯度——这是 stage 2 要消掉的部分。）"""
        if self.world_size > 1:
            dist.all_reduce(flat_grad)
            flat_grad /= self.world_size
        return flat_grad[self.start:self.start + self.shard_size]

    def _sync_params(self) -> None:
        """把各 rank 更新后的参数分片 all-gather 回完整参数向量。"""
        if self.world_size > 1:
            self.flat.copy_(all_gather_shards(self.shard, self.world_size))

    def train_step(self, features: torch.Tensor, labels: torch.Tensor) -> float:
        self.model.zero_grad(set_to_none=True)
        loss = F.cross_entropy(self.model(features), labels)
        loss.backward()
        grad_shard = self._sync_grads(self._flatten_grads())
        self.opt.step([grad_shard])   # 只更新本 rank 的分片（in-place 作用于 flat）
        self._sync_params()
        return loss.item()

    def memory_report(self) -> dict:
        return {
            "params": self.total,          # 参数全量
            "grads": self.flat.numel(),    # 梯度全量（含补齐段）
            "optim": self.opt.state_numel(),
        }

    def final_flat_params(self) -> torch.Tensor:
        return self.flat[:self.total].clone()
