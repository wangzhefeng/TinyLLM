# -*- coding: utf-8 -*-

# ***************************************************
# * File        : linear_base.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-07-28
# * Version     : 1.0.072810
# * Description : 并行线性层的基类
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

__all__ = []

# python libraries
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributed as dist

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


class LinearBase(nn.Module):
    """
    LinearBase 是一个抽象基类(nn.Module), 它为所有并行的线性层提供了基础框架和通用属性
    """
    
    def __init__(self, input_size: int, output_size: int, tp_dim: int | None=None):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.tp_dim = tp_dim  # 张量并行的维度(0: column, 1: row)
        self.tp_rank = dist.get_rank()  # 当前 GPU 的 rank
        self.tp_size = dist.get_world_size()  # 并行组的大小(GPU 数量)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class ColumnParallelLinear(LinearBase):
    """
    列并行
    """
    
    def __init__(self, input_size: int, output_size: int, bias: bool=False):
        super().__init__(input_size, output_size, 0)

        self.input_size_per_partition = input_size  # TODO
        self.output_size_per_partition = torch.div(output_size, self.tp_size)
        
        self.weight = nn.Parameter(torch.empty(self.output_size_per_partition, self.input_size))
        self.weight.weight_loader = self.weight_loader
        
        if bias:
            self.bias = nn.Parameter(torch.empty(self.output_size_per_partition))
            self.bias.weighg_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(self.tp_dim)  # self.tp_dim = 0
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输入 x 是完整的，未经切分
        x = F.linear(x, self.weight, self.bias)


class RowParallelLinear(LinearBase):
    """
    行并行
    """
    
    def __init__(self, input_size: int, output_size: int, bias: bool=False):
        super().__init__(input_size, output_size, 1)

        self.input_size_per_partition = torch.div(input_size, self.tp_size)
        self.output_size_per_partition = output_size  # TODO

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
        # 输入 x 是完整的，未经切分
        # bias 只在第一个 GPU(tp_rank=0) 上添加，因为 all_reduce 会将所有 GPU 的结果相加，如果不做处理，bias 会被添加 tp_size 次
        y = F.linear(
            x, 
            self.weight, 
            self.bias if self.tp_rank == 0 else None
        )
        if self.tp_size > 1:
            dist.all_reduce(y)  # 对所有 GPU 结果求和
        
        return y
 

class QKVParallelLinear(ColumnParallelLinear):
    
    def __init__(self, hidden_size: int, head_size: int, total_num_heads: int, total_num_kv_heads: int|None=None, bias: bool=False):
        
        input_size = None
        output_size = None
        bias = None

        # 调用父类 ColumnParallelLinear 的初始化方法
        super().__init__(input_size, output_size, bias)


# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
