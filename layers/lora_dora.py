# -*- coding: utf-8 -*-

# ***************************************************
# * File        : lora.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-02-21
# * Version     : 0.1.022123
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.log_util import logger

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
        super().__init__()

        # A
        # version 1
        # self.A = nn.Parameter(torch.empty(in_dim, rank))
        # nn.init.kaiming_uniform_(self.A, a = math.sqrt(5))
        # version 2
        std_dev = 1 / torch.sqrt(torch.tensor(rank).float())
        self.A = nn.Parameter(torch.randn(in_dim, rank) * std_dev)
        # B
        # version 1
        # self.B = nn.Parameter(torch.empty(rank, out_dim))
        # version 2
        self.B = nn.Parameter(torch.zeros(rank, out_dim))
        # alpha
        self.alpha = alpha
    
    def forward(self, x):
        x = self.alpha * (x @ self.A @ self.B)

        return x


class LinearWithLoRA(nn.Module):
    """
    与 LinearWithLoRAMerged 等价
    """
    
    def __init__(self, linear, rank, alpha):
        super().__init__()

        self.linear = linear
        self.lora = LoRALayer(
            linear.in_features, 
            linear.out_features, 
            rank, 
            alpha,
        )
    
    def forward(self, x):
        x = self.linear(x) + self.lora(x)
        
        return x


class LinearWithDoRA(nn.Module):
    """
    与 LinearWithDoRAMerged 等价
    """
    
    def __init__(self, linear, rank, alpha):
        super().__init__()
        
        self.linear = linear
        self.lora = LoRALayer(
            linear.in_features, 
            linear.out_features, 
            rank, 
            alpha
        )
        self.m = nn.Parameter(
            torch.ones(1, linear.out_features)
        )

    def forward(self, x):
        linear_output = self.linear(x)
        lora_output = self.lora(x)
        lora_output_norm = lora_output / (lora_output.norm(p=2, dim=1, keepdim=True) + 1e-9)
        dora_modification = self.m * lora_output_norm
        
        return linear_output + dora_modification


class LinearWithLoRAMerged(nn.Module):
    """
    This LoRA code is equivalent to LinearWithLoRA
    """

    def __init__(self, linear, rank, alpha) -> None:
        super().__init__()

        self.linear = linear
        self.lora = LoRALayer(
            linear.in_features, 
            linear.out_features, 
            rank, 
            alpha,
        )

    def forward(self, x):
        # combine LoRA matrices
        lora = self.lora.A @ self.lora.B
        # combine LoRA with original weights
        combined_weight = self.linear.weight + self.lora.alpha * lora.T

        output = F.linear(x, combined_weight, self.linear.bias)

        return output


class LinearWithDoRAMerged(nn.Module):
    """
    Code inspired by https://github.com/catid/dora/blob/main/dora.py
    DoRA 可以用两个步骤来描述：
        - 第一步是将预训练的权重矩阵分解为一个幅度向量（m）和一个方向矩阵（V）
        - 第二步是对方向矩阵 V 应用 LoRA，并单独训练幅度向量 m。
    """

    def __init__(self, linear, rank, alpha) -> None:
        super().__init__()

        self.linear = linear
        self.lora = LoRALayer(
            linear.in_features, 
            linear.out_features, 
            rank, 
            alpha,
        )
        # magnitude vector(幅度向量, 一个标量值，表示长度)
        self.m = nn.Parameter(
            self.linear.weight.norm(p=2, dim=0, keepdim=True)
        )

    def forward(self, x):
        # combine LoRA matrices
        lora = self.lora.A @ self.lora.B
        # combine LoRA with original weights
        combined_weight = self.linear.weight + self.lora.alpha * lora.T
        # directional matrix(方向矩阵, 一个单位向量，表示其在空间中的取向)
        column_norm = combined_weight.norm(p=2, dim=0, keepdim=True)
        V = combined_weight / column_norm
        # 任何向量都可以表示为其幅度与其方向的乘积,DoRA中，将大小和方向分解应用于整个预训练权重
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
