# -*- coding: utf-8 -*-

# ***************************************************
# * File        : mlp.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-07-27
# * Version     : 1.1.072816
# * Description : Qwen3MLP 前馈神经网络
# *               ├── gate_up_proj: MergedColumnParallelLinear (gate/up 打包)
# *               ├── act_fn: SiluAndMul (silu(gate) * up)
# *               └── down_proj: RowParallelLinear
# * Link        : https://github.com/GeeeekExplorer/nano-vllm
# * Requirement : torch >= 2.0
# ***************************************************

__all__ = [
    "SiluAndMul",
    "Qwen3MLP",
]

# python libraries
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F

from .parallel_linear import MergedColumnParallelLinear, RowParallelLinear

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


class SiluAndMul(nn.Module):
    """
    SwiGLU 激活: silu(gate) * up，输入最后一维为 [gate | up] 拼接
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = x.chunk(2, dim=-1)
        return F.silu(gate) * up


class Qwen3MLP(nn.Module):
    """
    前馈神经网络
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str = "silu",
    ):
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
        )
        assert hidden_act == "silu"
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x = self.down_proj(x)
        return x




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
