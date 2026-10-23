# -*- coding: utf-8 -*-

# ***************************************************
# * File        : zero3.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2026-10-24
# * Version     : 1.0.102401
# * Description : 03 单元 ZeRO stage 3：优化器状态 + 梯度 + 参数全分片
# *               （P_os+g+p）。每个参数只有一个 owner rank 持久持有真值，
# *               其它 rank 只持有 0 元素的空壳；forward 时按层 hook 从
# *               owner broadcast 物化权重，backward 后把梯度 reduce 给
# *               owner 并立即释放非 owner 的权重与梯度。
# *               显存账：参数、梯度、优化器状态全部 ~N/w。
# *               简化点（见 ZeroThreeTrainer docstring）：只支持 ToyMLP
# *               这类"所有参数都参与每步反向"的简单结构。
# * Link        : https://arxiv.org/abs/1910.02054
# * Requirement : torch >= 2.5；gloo（CPU）或 nccl（CUDA）
# ***************************************************

__all__ = ["ZeroThreeTrainer"]

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
import torch.nn as nn
import torch.nn.functional as F

from zero_common import ShardedAdam


def assign_owners(params: list[torch.Tensor], world_size: int) -> list[int]:
    """
    按参数大小贪心均衡地把每个参数整体分配给一个 owner rank
    （按 numel 降序，依次放到当前负载最小的 rank）。

    教学简化：工业实现（ZeRO-3/FSDP）把参数拼平后按元素切分，单个参数
    可能横跨多个 rank；这里以"整个参数"为最小分片单位，换来物化逻辑
    （从 owner broadcast 整个参数）的直观。ToyMLP 参数少，负载均衡
    不完美，实测数字见训练末尾的显存账表。
    """
    order = sorted(range(len(params)), key=lambda i: -params[i].numel())
    loads = [0] * world_size
    owners = [0] * len(params)
    for i in order:
        r = loads.index(min(loads))
        owners[i] = r
        loads[r] += params[i].numel()
    return owners


class ZeroThreeTrainer:
    """
    ZeRO stage 3：参数本体也分片。

    每步流程：
      1. 逐层物化：每个 nn.Linear 的 forward_pre_hook 从 owner broadcast
         该层 weight/bias（非 owner 此时才分配完整存储）；
      2. 前向/反向在物化的完整参数上进行；
      3. 逐参数把梯度 reduce 到 owner 并求平均，非 owner 立即丢弃梯度；
      4. owner 在自己拥有的参数上执行 ShardedAdam 更新；
      5. 释放非 owner 的参数存储（回到 0 元素空壳），等待下一步重新物化。

    简化点（相对 FSDP/ZeRO-3 工业实现）：
      - 物化的权重在整个反向期间一直保留；FSDP 的 reshard_after_forward
        会在前向结束后立即释放、反向前再 all-gather 一次，峰值更低；
      - 没有通信/计算重叠与预取（prefetch），按层串行 broadcast；
      - 以整个参数为分片单位（见 assign_owners），不做参数内部切分；
      - 假设所有参数每步都参与反向（ToyMLP 满足），不处理部分冻结。
    FSDP2 就是这套思想的工业实现，见下一单元 04-fsdp。
    """

    def __init__(self, model: torch.nn.Module, rank: int, world_size: int, lr: float):
        self.model = model
        self.rank = rank
        self.world_size = world_size
        self.params = list(model.parameters())
        self.total = sum(p.numel() for p in self.params)
        self.owners = assign_owners(self.params, world_size)
        self.shape_of = {id(p): tuple(p.shape) for p in self.params}
        self.owner_of = {id(p): o for p, o in zip(self.params, self.owners)}

        # 非 owner 立即释放参数存储，只留空壳
        for p, o in zip(self.params, self.owners):
            if o != rank:
                p.data = torch.empty(0)
        self.owned = [p for p, o in zip(self.params, self.owners) if o == rank]
        self.opt = ShardedAdam(self.owned, lr=lr)

        # 逐层物化 hook：forward 走到哪一层，就先 broadcast 哪一层的权重
        for m in model.modules():
            if isinstance(m, nn.Linear):
                m.register_forward_pre_hook(self._materialize_hook)

    def _materialize_hook(self, module: nn.Module, _inputs) -> None:
        with torch.no_grad():
            for p in module.parameters(recurse=False):
                owner = self.owner_of[id(p)]
                if p.data.numel() == 0:
                    p.data = torch.empty(self.shape_of[id(p)])
                if self.world_size > 1:
                    dist.broadcast(p.data, src=owner)

    def train_step(self, features: torch.Tensor, labels: torch.Tensor) -> float:
        self.model.zero_grad(set_to_none=True)
        loss = F.cross_entropy(self.model(features), labels)  # hook 逐层物化
        loss.backward()

        # 梯度归 owner：逐参数 reduce 到 owner，非 owner 立即释放
        owned_grads = []
        for p, o in zip(self.params, self.owners):
            assert p.grad is not None, "存在未参与反向的参数，本教学实现不支持"
            if self.world_size > 1:
                dist.reduce(p.grad, dst=o)
            if o == self.rank:
                if self.world_size > 1:
                    p.grad /= self.world_size
                owned_grads.append(p.grad)
            else:
                p.grad = None

        # owner 只在自己拥有的参数上更新
        self.opt.step(owned_grads)

        # 释放非 owner 的参数存储，回到 1/w 持久占用
        for p, o in zip(self.params, self.owners):
            if o != self.rank:
                p.data = torch.empty(0)
        return loss.item()

    def memory_report(self) -> dict:
        owned_numel = sum(p.numel() for p in self.owned)
        return {
            "params": owned_numel,   # 参数只持久持有 owner 部分（物化是瞬时的）
            "grads": owned_numel,    # 梯度只保留 owner 部分
            "optim": self.opt.state_numel(),
        }

    def final_flat_params(self) -> torch.Tensor:
        """从各 owner broadcast 出完整参数并拼平（用于等价性校验）。"""
        with torch.no_grad():
            for p, o in zip(self.params, self.owners):
                if p.data.numel() == 0:
                    p.data = torch.empty(self.shape_of[id(p)])
                if self.world_size > 1:
                    dist.broadcast(p.data, src=o)
            return torch.cat([p.detach().reshape(-1) for p in self.params])
