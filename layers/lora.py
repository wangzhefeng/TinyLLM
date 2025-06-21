# -*- coding: utf-8 -*-

# ***************************************************
# * File        : lora.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2025-02-21
# * Version     : 0.1.022123
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import os
import sys
ROOT = str(os.getcwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)
import math
from pathlib import Path

import torch
import torch.nn as nn

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


class LoRALayer(nn.Module):
    """
    初始化一个 LoRA Layer，它会创建矩阵 A 和 B，同时设置:  
        - 秩超参数 rank(r): rank 是一个超参数，用于控制矩阵 A和 B 的内维度。
          换句话说，该参数决定了 LoRA 引入的额外参数数量，是平衡模型适应性和参数效率的关键因素
        - 缩放超参数 alpha: alpha 是一个缩放超参数，作用于低秩适配的输出。
          它主要控制适配层输出对原始层输出的影响程度，可视为调节低秩适配对层输出影响的一种方式
    """
    
    def __init__(self, in_dim, out_dim, rank, alpha):
        super(LoRALayer, self).__init__()

        # A
        self.A = nn.Parameter(torch.empty(in_dim, rank))
        nn.init.kaiming_uniform_(self.A, a = math.sqrt(5))
        # B
        self.B = nn.Parameter(torch.empty(rank, out_dim))
        # alpha
        self.alpha = alpha
    
    def forward(self, x):
        x = self.alpha * (x @ self.A @ self.B)

        return x 


class LinearWithLoRA(nn.Module):
    
    def __init__(self, linear, rank, alpha):
        super(LinearWithLoRA, self).__init__()

        self.linear = linear
        self.lora = LoRALayer(
            linear.in_features, 
            linear.out_features, 
            rank, 
            alpha
        )
    
    def forward(self, x):
        return self.linear(x) + self.lora(x)


def replace_linear_with_lora(model, rank, alpha):
    """
    将模型中的所有 Linear 层替换为新的 LinearWithLoRA 层
    """
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            # 用 LinearWithLoRA 替换先前的 nn.Linear
            setattr(model, name, LinearWithLoRA(module, rank, alpha))
        else:
            # 对子模块递归调用
            replace_linear_with_lora(module, rank, alpha)




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
