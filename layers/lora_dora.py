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
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]

from utils.log_util import logger


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


class LoRALayer_v2(nn.Module):
    """
    初始化一个 LoRA Layer，它会创建矩阵 A 和 B，同时设置:  
        - 秩超参数 rank(r): rank 是一个超参数，用于控制矩阵 A和 B 的内维度。
          换句话说，该参数决定了 LoRA 引入的额外参数数量，是平衡模型适应性和参数效率的关键因素
        - 缩放超参数 alpha: alpha 是一个缩放超参数，作用于低秩适配的输出。
          它主要控制适配层输出对原始层输出的影响程度，可视为调节低秩适配对层输出影响的一种方式
    """
    
    def __init__(self, in_dim, out_dim, rank, alpha):
        super(LoRALayer_v2, self).__init__()

        # A
        std_dev = 1 / torch.sqrt(torch.tensor(rank).float())
        self.A = nn.Parameter(torch.randn(in_dim, rank) * std_dev)
        # B
        self.B = nn.Parameter(torch.zeros(rank, out_dim))
        # alpha
        self.alpha = alpha
    
    def forward(self, x):
        x = self.alpha * (x @ self.A @ self.B)

        return x


class LinearWithLoRA(nn.Module):
    
    def __init__(self, linear, rank, alpha):
        super(LinearWithLoRA, self).__init__()

        self.linear = linear
        self.lora = LoRALayer_v2(
            linear.in_features, 
            linear.out_features, 
            rank, 
            alpha,
        )
    
    def forward(self, x):
        x = self.linear(x) + self.lora(x)
        
        return x


class LinearWithLoRAMerged(nn.Module):
    """
    This LoRA code is equivalent to LinearWithLoRA
    """

    def __init__(self, linear, rank, alpha) -> None:
        super(LinearWithLoRAMerged, self).__init__()

        self.linear = linear
        self.lora = LoRALayer_v2(
            linear.in_features, 
            linear.out_features, 
            rank, 
            alpha,
        )

    def forward(self, x):
        lora = self.lora.A @ self.lora.B
        combined_weight = self.linear.weight + lora.T
        output = F.linear(x, combined_weight, self.linear.bias)

        return output


class LinearWithDoRAMerged(nn.Module):
    """
    Code inspired by https://github.com/catid/dora/blob/main/dora.py
    """

    def __init__(self, linear, rank, alpha) -> None:
        super(LinearWithDoRAMerged, self).__init__()

        self.linear = linear
        self.lora = LoRALayer_v2(
            linear.in_features, 
            linear.out_features, 
            rank, 
            alpha,
        )
        self.m = nn.Parameter(
            self.linear.weight.norm(p=2, dim=0, keepdim=True)
        )

    def forward(self, x):
        lora = self.lora.A @ self.lora.B
        combined_weight = self.linear.weight + self.lora.alpha * lora.T
        column_norm = combined_weight.norm(p=2, dim=0, keepdim=True)
        V = combined_weight / column_norm
        new_weight = self.m * V
        output = F.linear(x, new_weight, self.linear.bias)

        return output


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


def freeze_linear_layers(model, verbose=False):
    """
    将模型中的所有 Linear 层替换为新的 LinearWithLoRA 层
    """
    for child in model.children():
        if isinstance(child, nn.Linear):
            for param in child.parameters():
                param.requires_grad = False
        else:
            # Recursively freeze linear layers in children modules
            freeze_linear_layers(child)
    # Check if linear layers are frozen
    if verbose:
        logger.info("Check if linear layers are frozen:")
        for name, param in model.named_parameters():
            logger.info(f"{name}: {param.requires_grad}")

    



# 测试代码 main 函数
def main():
    torch.manual_seed(123)
 
    # input
    x = torch.randn((1, 10))

    # linear
    linear = nn.Linear(10, 2)
    # original output
    logger.info(f"original output: {linear(x)}")
    # lora output
    layer_lora_1 = LinearWithLoRA(linear, rank=2, alpha=4)
    logger.info(f"lora output: {layer_lora_1(x)}")
    # lora output
    layer_lora_2 = LinearWithLoRAMerged(linear, rank=2, alpha=4)
    logger.info(f"lora output: {layer_lora_2(x)}")
    # dora output
    layer_dora = LinearWithDoRAMerged(linear, rank=2, alpha=4)
    logger.info(f"dora output: {layer_dora(x)}")

if __name__ == "__main__":
    main()
