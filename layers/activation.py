# -*- coding: utf-8 -*-

# ***************************************************
# * File        : activation.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-01-25
# * Version     : 1.0.012520
# * Description : description
# * Link        : SiLU: https://arxiv.org/abs/1702.03118
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# * TODO        : 1.
# ***************************************************

__all__ = [
    "GELU",
    "SiLU",
]

# python libraries
import os
import sys
ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import torch
import torch.nn as nn

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


class GELU(nn.Module):
    
    def __init__(self):
        super(GELU, self).__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) * 
            (x + 0.044715 * torch.pow(x, 3))
        ))


class ReLU(nn.Module):
    
    def __init__(self):
        super(ReLU, self).__init__()

    def forward(self, x):
        return torch.maximum(torch.tensor(0.0), x)


class SiLU(nn.Module):
    
    def __init__(self):
        super(SiLU, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)




# 测试代码 main 函数
def main():
    import matplotlib.pyplot as plt

    # func
    gelu = GELU()
    # gelu = nn.GELU() 
    relu = ReLU()
    # relu = nn.ReLU() 
    silu = SiLU()
    # silu = nn.SiLU()

    # data
    x = torch.linspace(-3, 3, 100)
    y_gelu, y_relu, y_silu = gelu(x), relu(x), silu(x)

    # plot
    plt.figure(figsize=(12, 3))
    for i, (y, label) in enumerate(zip([y_gelu, y_relu, y_silu], ["GELU", "ReLU", "SiLU"]), 1):
        plt.subplot(1, 3, i)
        plt.plot(x, y)
        plt.title(f"{label} activation function")
        plt.xlabel("x")
        plt.ylabel(f"{label}(x)")
        plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
