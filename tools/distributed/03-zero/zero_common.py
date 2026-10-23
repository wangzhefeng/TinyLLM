# -*- coding: utf-8 -*-

# ***************************************************
# * File        : zero_common.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2026-10-24
# * Version     : 1.0.102401
# * Description : 03 单元 ZeRO 各 stage 共用的基础设施：
# *               - ShardedAdam：手写 Adam，只在"本 rank 拥有的参数分片"
# *                 上维护 exp_avg / exp_avg_sq 并执行更新，不依赖
# *                 torch.optim.Adam 的内部状态布局，分片逻辑完全透明；
# *               - 拼平/分片辅助与手写集合通信（reduce-scatter、
# *                 all-gather 的 gloo 安全等价写法）；
# *               - FullReplicaTrainer：stage 0 基线（DDP 等价的全量复制）；
# *               - 各 stage 的"显存账"对比表打印与等价性比对辅助。
# * Link        : https://arxiv.org/abs/1910.02054
# * Requirement : torch >= 2.5；gloo（CPU）或 nccl（CUDA）
# ***************************************************

__all__ = [
    "ShardedAdam",
    "FullReplicaTrainer",
    "pad_to_multiple",
    "reduce_scatter_avg",
    "all_gather_shards",
    "print_memory_report",
    "verify_against_ref",
]

# python libraries
import sys
from pathlib import Path

# repo root: tools/distributed/03-zero/zero_common.py -> parents[3]
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import torch
import torch.distributed as dist
import torch.nn.functional as F

from tools.distributed._common.dist import is_main_process, rank0_print

# ZeRO 论文（arXiv:1910.02054）中三阶段的分片对象，按"参数/梯度/优化器状态"列出。
# stage 1 只分片优化器状态；stage 2 加分片梯度；stage 3 再加分片参数。
THEORY_RATIOS = {
    0: ("1", "1", "1"),
    1: ("1", "1", "1/w"),
    2: ("1", "1/w", "1/w"),
    3: ("1/w", "1/w", "1/w"),
}


class ShardedAdam:
    """
    手写 Adam，优化器状态（exp_avg/exp_avg_sq）与传入的参数分片同形。

    调用方只把"本 rank 拥有的分片"传进来，因此状态天然就是分片的；
    数学上与 torch.optim.Adam 完全一致（逐元素更新 + 全局标量步数 t），
    所以无论把参数向量切成几段、按什么顺序切，结果都与整向量 Adam 相同——
    这正是 verify_equivalence.py 能与单进程基线逐位比对的基础。
    """

    def __init__(self, shards: list[torch.Tensor], lr: float, betas=(0.9, 0.999), eps=1e-8):
        self.shards = list(shards)
        self.exp_avg = [torch.zeros_like(s) for s in self.shards]
        self.exp_avg_sq = [torch.zeros_like(s) for s in self.shards]
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.t = 0

    @torch.no_grad()
    def step(self, grads: list[torch.Tensor]) -> None:
        assert len(grads) == len(self.shards), "grads 与 shards 必须一一对应"
        self.t += 1
        bias_c1 = 1 - self.b1 ** self.t
        bias_c2 = 1 - self.b2 ** self.t
        for g, p, m, v in zip(grads, self.shards, self.exp_avg, self.exp_avg_sq):
            m.mul_(self.b1).add_(g, alpha=1 - self.b1)
            v.mul_(self.b2).addcmul_(g, g, value=1 - self.b2)
            m_hat = m / bias_c1
            v_hat = v / bias_c2
            p.addcdiv_(m_hat, v_hat.sqrt().add_(self.eps), value=-self.lr)

    def state_numel(self) -> int:
        """优化器状态总元素数（m + v），用于显存账实测。"""
        return sum(s.numel() for s in self.exp_avg) + sum(s.numel() for s in self.exp_avg_sq)


def pad_to_multiple(n: int, m: int) -> int:
    """把总元素数补齐到 world_size 的整数倍，使各 rank 分片等长
    （等长分片可直接用 all_gather 列表接口拼接，且手写 reduce-scatter 切片规整）。
    补齐段的梯度恒为 0，Adam 对其更新恒为 0，不影响数学结果。"""
    return ((n + m - 1) // m) * m


def reduce_scatter_avg(flat_grad: torch.Tensor, rank: int, world_size: int, shard_size: int) -> torch.Tensor:
    """
    手写 reduce-scatter + 平均：第 r 段归约（sum）到 rank r，
    每个 rank 只取回并保留自己负责分片的平均梯度。

    生产实现用融合原语 dist.reduce_scatter_tensor；这里逐段 dist.reduce
    效果等价且 gloo 一定支持（非 dst 上分片内容未定义，但没人读它）。
    """
    for r in range(world_size):
        dist.reduce(flat_grad[r * shard_size:(r + 1) * shard_size], dst=r)
    return flat_grad[rank * shard_size:(rank + 1) * shard_size] / world_size


def all_gather_shards(my_shard: torch.Tensor, world_size: int) -> torch.Tensor:
    """
    手写 all-gather：把各 rank 等长的参数分片拼回完整向量。

    生产实现用融合原语 dist.all_gather_into_tensor；列表接口在所有后端可用。
    """
    shards = [torch.empty_like(my_shard) for _ in range(world_size)]
    dist.all_gather(shards, my_shard.contiguous())
    return torch.cat(shards)


class FullReplicaTrainer:
    """
    stage 0：DDP 等价基线。每个 rank 持有完整的参数、梯度与优化器状态，
    反向后对梯度 all-reduce 求平均，各自在完整参数上执行相同的 Adam 更新
    （各 rank 更新完全冗余，这正是 ZeRO 要逐步消除的部分）。
    """

    def __init__(self, model: torch.nn.Module, rank: int, world_size: int, lr: float):
        self.model = model
        self.rank = rank
        self.world_size = world_size
        self.params = list(model.parameters())
        self.opt = ShardedAdam(self.params, lr=lr)

    def train_step(self, features: torch.Tensor, labels: torch.Tensor) -> float:
        self.model.zero_grad(set_to_none=True)
        loss = F.cross_entropy(self.model(features), labels)
        loss.backward()
        for p in self.params:
            if self.world_size > 1:
                dist.all_reduce(p.grad)
                p.grad /= self.world_size
        self.opt.step([p.grad for p in self.params])
        return loss.item()

    def memory_report(self) -> dict:
        n = sum(p.numel() for p in self.params)
        return {"params": n, "grads": n, "optim": self.opt.state_numel()}

    def final_flat_params(self) -> torch.Tensor:
        return torch.cat([p.detach().reshape(-1) for p in self.params])


def print_memory_report(stage: int, world_size: int, report: dict) -> None:
    """
    rank0 打印"显存账"对比表：ZeRO 论文的理论分片比例 vs 本 rank 实际
    持有的元素数（numel）。CPU 无显存计数器，numel × dtype 字节数即占用，
    对 fp32 乘 4 就是字节数。
    """
    vec = torch.tensor([report["params"], report["grads"], report["optim"]], dtype=torch.long)
    if world_size > 1:
        gathered = [torch.zeros_like(vec) for _ in range(world_size)]
        dist.all_gather(gathered, vec)
    else:
        gathered = [vec]
    if not is_main_process():
        return
    t_param, t_grad, t_optim = THEORY_RATIOS[stage]
    print(f"\n[stage {stage}] 每 rank 显存账（单位: numel，fp32 乘 4 得字节数；w={world_size}）")
    print(f"  {'rank':>4} | {'参数 实测(理论)':>16} | {'梯度 实测(理论)':>16} | {'优化器状态 实测(理论)':>20}")
    for r, v in enumerate(gathered):
        print(f"  {r:>4} | {v[0].item():>7d} ({t_param:>3})   | {v[1].item():>7d} ({t_grad:>3})   | {v[2].item():>7d} ({t_optim:>3})")
    if stage >= 2:
        print("  注：反向传播瞬间 autograd 仍会临时物化全量梯度（教学简化）；")
        print("      生产实现（DeepSpeed/FSDP）用 backward hook 分桶归约后即时释放。")


def verify_against_ref(flat: torch.Tensor, last_loss: float, ref_path: str,
                       rtol=1e-5, atol=1e-6, loss_tol=1e-5) -> tuple[bool, str]:
    """
    与单进程基线比对：最终参数向量 allclose + 最终 loss 差在容差内。
    所有 rank 执行同样的比对（flat 经集合通信后各 rank 一致），返回相同的结论。
    """
    ref = torch.load(ref_path, map_location="cpu", weights_only=True)
    ref_flat = ref["flat"]
    max_diff = (flat - ref_flat).abs().max().item()
    loss_diff = abs(last_loss - ref["losses"][-1])
    ok = torch.allclose(flat, ref_flat, rtol=rtol, atol=atol) and loss_diff < loss_tol
    msg = f"max|Δparam|={max_diff:.3e}, |Δloss|={loss_diff:.3e}"
    return ok, msg
