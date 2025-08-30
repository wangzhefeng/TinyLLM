# -*- coding: utf-8 -*-

# ***************************************************
# * File        : activation.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-01-25
# * Version     : 1.0.012520
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

__all__ = [
    "ReLU",
    "ReLUPyTorch",
    "GELU",
    "SiLU",
]

# python libraries
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)

import torch
import torch.nn as nn

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


class ReLU(nn.Module):
    """
    ReLU (Rectified Linear Unit, 整流线性单元)
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.maximum(torch.tensor(0.0), x)


class ReLUPyTorch(nn.Module):
    """
    ReLU (Rectified Linear Unit, 整流线性单元)
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return nn.functional.relu(x)


class GELU(nn.Module):
    """
    GELU(Gaussian Error Linear Unit, 高斯误差线性单元): https://arxiv.org/abs/1606.08415
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) * 
            (x + 0.044715 * torch.pow(x, 3))
        ))


class SiLU(nn.Module):
    """
    SiLU: https://arxiv.org/abs/1702.03118
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


# TODO
class SwiGLU(nn.Module):
    """
    SwiGLU(Swish-Gated Linear Unit, Swish 门控线性单元):  
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass




# 测试代码 main 函数
def main():
    import matplotlib.pyplot as plt

    # func
    relu = ReLU()
    # relu = nn.ReLU() 

    relu_torch = ReLUPyTorch()
    # relu_torch = nn.functional.relu()

    gelu = GELU()
    # gelu = nn.GELU() 
    
    silu = SiLU()
    # silu = nn.SiLU()
    # silu = nn.functional.silu()

    # data
    x = torch.linspace(-3, 3, 100)
    y_relu, y_relu_torch, y_gelu, y_silu = relu(x), relu_torch(x), gelu(x), silu(x)
    
    # plot
    plt.figure(figsize=(12, 3))
    for i, (y, label) in enumerate(zip([y_relu, y_relu_torch, y_gelu, y_silu], ["ReLU", "ReLU_torch", "GELU", "SiLU"]), 1):
        plt.subplot(1, 4, i)
        plt.plot(x, y)
        plt.title(f"{label} activation function")
        plt.xlabel("x")
        plt.ylabel(f"{label}(x)")
        plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
