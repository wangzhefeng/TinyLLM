# -*- coding: utf-8 -*-

# ***************************************************
# * File        : zero2.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2026-10-24
# * Version     : 1.0.102401
# * Description : 03 单元 ZeRO stage 2：优化器状态 + 梯度分片（P_os+g）。
# *               相对 stage 1 的增量：梯度不再做全量 all-reduce，
# *               改用 reduce-scatter——第 r 段梯度只归约到 rank r，
# *               每个 rank 只保留自己负责分片的平均梯度（1/w），
# *               更新分片后照常 all-gather 参数。
# *               显存账：优化器状态 2N/w、梯度 N → N/w（持久部分）、
# *               参数 N 不变。
# * Link        : https://arxiv.org/abs/1910.02054
# * Requirement : torch >= 2.5；gloo（CPU）或 nccl（CUDA）
# ***************************************************

__all__ = ["ZeroTwoTrainer"]

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

from zero_common import reduce_scatter_avg
from zero1 import ZeroOneTrainer


class ZeroTwoTrainer(ZeroOneTrainer):
    """
    ZeRO stage 2 = stage 1 + 梯度分片。

    与 stage 1 唯一的差别在 _sync_grads：all-reduce（每个 rank 拿到全量
    平均梯度）换成 reduce-scatter（每个 rank 只拿到自己分片的平均梯度）。
    对不属于本 rank 的分片，梯度算完即弃，持久梯度存储降到 1/w。

    教学简化说明：autograd 反向传播结束后所有 p.grad 仍会一次性物化
    （瞬时峰值仍是全量 N）；生产实现（DeepSpeed/FSDP）注册 backward
    hook，把梯度按桶（bucket）归约后立即释放，峰值也压到 ~一桶 + N/w。
    本实现关注的是"持久占用"这一 ZeRO 论文的记账口径。
    """

    def _sync_grads(self, flat_grad: torch.Tensor) -> torch.Tensor:
        if self.world_size == 1:
            return flat_grad[self.start:self.start + self.shard_size]
        return reduce_scatter_avg(flat_grad, self.rank, self.world_size, self.shard_size)

    def memory_report(self) -> dict:
        return {
            "params": self.total,           # 参数全量
            "grads": self.shard_size,       # 梯度只保留 1/w 分片
            "optim": self.opt.state_numel(),
        }
