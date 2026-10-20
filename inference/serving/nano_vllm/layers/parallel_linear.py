# -*- coding: utf-8 -*-

# ***************************************************
# * File        : parallel_linear.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-07-28
# * Version     : 1.1.072816
# * Description : 张量并行(TP)线性层。
# *               ├── LinearBase: 并行层抽象基类
# *               ├── ColumnParallelLinear: 列并行(按输出维度切分)
# *               ├── RowParallelLinear: 行并行(按输入维度切分 + all-reduce)
# *               ├── QKVParallelLinear: Q/K/V 打包的列并行层
# *               └── MergedColumnParallelLinear: 多个输出拼接的列并行层(gate_up_proj)
# *               注意：torch.distributed 未初始化时自动退化为 world_size=1
# *               的单进程语义，即普通线性层。
# * Link        : https://github.com/GeeeekExplorer/nano-vllm
# * Requirement : torch >= 2.0
# ***************************************************

__all__ = [
    "LinearBase",
    "ReplicatedLinear",
    "ColumnParallelLinear",
    "RowParallelLinear",
    "QKVParallelLinear",
    "MergedColumnParallelLinear",
]

# python libraries
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)
PKG_ROOT = str(Path(__file__).resolve().parents[1])
if PKG_ROOT in sys.path:
    sys.path.remove(PKG_ROOT)
sys.path.insert(0, PKG_ROOT)
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributed as dist

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


def _tp_rank_and_size() -> tuple[int, int]:
    """
    获取 TP rank / world_size。
    torch.distributed 未初始化时退化为 (0, 1)，即单进程普通线性层语义。
    """
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def _divide(numerator: int, denominator: int) -> int:
    assert numerator % denominator == 0, f"{numerator} 无法被 {denominator} 整除"
    return numerator // denominator


class LinearBase(nn.Module):
    """
    LinearBase 是一个抽象基类(nn.Module)，为所有并行线性层提供基础框架和通用属性
    """

    def __init__(self, input_size: int, output_size: int, tp_dim: int | None = None):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.tp_dim = tp_dim  # 张量并行的维度(0: column, 1: row)
        self.tp_rank, self.tp_size = _tp_rank_and_size()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class ReplicatedLinear(LinearBase):
    """
    复制线性层：权重不切分，每个 TP rank 持有完整副本(对齐上游 layers/linear.py)
    """

    def __init__(self, input_size: int, output_size: int, bias: bool = False):
        super().__init__(input_size, output_size)

        self.weight = nn.Parameter(torch.empty(output_size, input_size))
        self.weight.weight_loader = self.weight_loader

        if bias:
            self.bias = nn.Parameter(torch.empty(output_size))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param.data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class ColumnParallelLinear(LinearBase):
    """
    列并行：按输出维度切分权重 [output_size, input_size] -> [output_size/tp, input_size]
    """

    def __init__(self, input_size: int, output_size: int, bias: bool = False):
        super().__init__(input_size, output_size, 0)

        self.input_size_per_partition = input_size
        self.output_size_per_partition = _divide(output_size, self.tp_size)

        self.weight = nn.Parameter(torch.empty(self.output_size_per_partition, self.input_size))
        self.weight.weight_loader = self.weight_loader

        if bias:
            self.bias = nn.Parameter(torch.empty(self.output_size_per_partition))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(self.tp_dim)  # self.tp_dim = 0
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输入 x 是完整的，未经切分；world_size=1 时即为普通线性层
        return F.linear(x, self.weight, self.bias)


class RowParallelLinear(LinearBase):
    """
    行并行：按输入维度切分权重 [output_size, input_size] -> [output_size, input_size/tp]
    """

    def __init__(self, input_size: int, output_size: int, bias: bool = False):
        super().__init__(input_size, output_size, 1)

        self.input_size_per_partition = _divide(input_size, self.tp_size)
        self.output_size_per_partition = output_size

        self.weight = nn.Parameter(torch.empty(self.output_size, self.input_size_per_partition))
        self.weight.weight_loader = self.weight_loader

        if bias:
            self.bias = nn.Parameter(torch.empty(self.output_size))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(self.tp_dim)  # self.tp_dim = 1
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输入 x 是当前分片的输入；bias 只在 tp_rank=0 上添加，
        # 否则 all_reduce 求和后 bias 会被加 tp_size 次
        y = F.linear(
            x,
            self.weight,
            self.bias if self.tp_rank == 0 else None,
        )
        if self.tp_size > 1:
            dist.all_reduce(y)  # 对所有 rank 的结果求和

        return y


class QKVParallelLinear(ColumnParallelLinear):
    """
    将 Q/K/V 三个投影打包为一个列并行层，输出布局为 [q_heads | k_heads | v_heads]
    """

    def __init__(self, hidden_size: int, head_size: int, total_num_heads: int, total_num_kv_heads: int | None = None, bias: bool = False):
        total_num_kv_heads = total_num_kv_heads or total_num_heads
        tp_size = _tp_rank_and_size()[1]

        self.head_size = head_size
        self.num_heads = _divide(total_num_heads, tp_size)
        self.num_kv_heads = _divide(total_num_kv_heads, tp_size)
        # 打包后的输出维度 = (num_q_heads + 2 * num_kv_heads) * head_size
        output_size = (total_num_heads + 2 * total_num_kv_heads) * head_size

        # 调用父类 ColumnParallelLinear 的初始化方法
        super().__init__(hidden_size, output_size, bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: str):
        """
        按 shard("q"/"k"/"v")分别加载打包权重。
        param 为当前 rank 的本地分片，shard_offset 是分片内偏移；
        全局 shard 权重按 tp_rank 切出本 rank 的部分(narrow)。
        """
        param_data = param.data
        assert loaded_shard_id in ("q", "k", "v")
        if loaded_shard_id == "q":
            shard_size = self.num_heads * self.head_size
            shard_offset = 0
        elif loaded_shard_id == "k":
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size
        else:  # "v"
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = (self.num_heads + self.num_kv_heads) * self.head_size
        # 从全局 shard 中切出当前 TP rank 的部分，写入本地分片的对应区间
        loaded_weight = loaded_weight.narrow(0, self.tp_rank * shard_size, shard_size)
        param_data[shard_offset: shard_offset + shard_size].copy_(loaded_weight)


class MergedColumnParallelLinear(ColumnParallelLinear):
    """
    将多个列并行层(如 gate_proj / up_proj)沿输出维度拼接为一个层
    """

    def __init__(self, input_size: int, output_sizes: list[int], bias: bool = False):
        self.output_sizes = output_sizes
        super().__init__(input_size, sum(output_sizes), bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: int):
        """
        按 shard 下标(0: gate, 1: up)分别加载打包权重
        """
        param_data = param.data
        shard_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size
        shard_size = self.output_sizes[loaded_shard_id] // self.tp_size
        loaded_weight = loaded_weight.narrow(0, self.tp_rank * shard_size, shard_size)
        param_data[shard_offset: shard_offset + shard_size].copy_(loaded_weight)




# 测试代码 main 函数
def main():
    # world_size=1(dist 未初始化) 时等价于普通线性层
    torch.manual_seed(0)
    x = torch.randn(3, 8)
    col = ColumnParallelLinear(8, 16, bias=True)
    ref = nn.Linear(8, 16, bias=True)
    with torch.no_grad():
        ref.weight.copy_(col.weight)
        ref.bias.copy_(col.bias)
    assert torch.allclose(col(x), ref(x))
    print("ColumnParallelLinear world_size=1 退化测试通过")

if __name__ == "__main__":
    main()
