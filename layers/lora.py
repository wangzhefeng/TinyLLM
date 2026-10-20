# -*- coding: utf-8 -*-

# ***************************************************
# * File        : lora.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2026-10-20
# * Version     : 1.0.102021
# * Description : LoRA / DoRA 参数高效微调层（LLMs-from-scratch 附录 E）。
# *               规范实现放在 layers/（供任意 models/ 模型即插即用）；
# *               post_training/peft/layers/lora_dora.py 为兼容 shim。
# * Link        : LoRA https://arxiv.org/abs/2106.09685
# *               DoRA https://arxiv.org/abs/2402.09353
# * Requirement : torch
# ***************************************************

__all__ = [
    "LoRALayer",
    "LinearWithLoRA",
    "LinearWithDoRA",
    "LinearWithLoRAMerged",
    "LinearWithDoRAMerged",
    "replace_linear_with_lora",
    "freeze_linear_layers",
    "merge_lora_weights",
]

# python libraries
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


class LoRALayer(nn.Module):
    """低秩适配 ΔW = alpha · (A @ B)，A: (in, rank)，B: (rank, out)。

    - rank(r)：低秩内维，控制新增参数量（参数效率 vs 适应能力的平衡）
    - alpha：缩放超参，控制适配输出对原始层输出的影响幅度
    """

    def __init__(self, in_dim, out_dim, rank, alpha):
        super().__init__()

        std_dev = 1 / torch.sqrt(torch.tensor(rank).float())
        self.A = nn.Parameter(torch.randn(in_dim, rank) * std_dev)
        self.B = nn.Parameter(torch.zeros(rank, out_dim))   # B 初始为零：ΔW 从零开始
        self.alpha = alpha

    def forward(self, x):
        return self.alpha * (x @ self.A @ self.B)


class LinearWithLoRA(nn.Module):
    """线性层 + LoRA 旁路（不合并版）：y = Wx + alpha·xAB。"""

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
        return self.linear(x) + self.lora(x)


class LinearWithDoRA(nn.Module):
    """线性层 + DoRA（不合并版）：把 LoRA 输出归一化后乘可学习幅度 m。"""

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
    """线性层 + LoRA（合并版）：推断时把 alpha·BA^T 加进权重，无旁路开销。"""

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
        lora = self.lora.A @ self.lora.B
        combined_weight = self.linear.weight + self.lora.alpha * lora.T

        return F.linear(x, combined_weight, self.linear.bias)


class LinearWithDoRAMerged(nn.Module):
    """线性层 + DoRA（合并版）：权重分解为幅度 m × 方向 V。"""

    def __init__(self, linear, rank, alpha) -> None:
        super().__init__()

        self.linear = linear
        self.lora = LoRALayer(
            linear.in_features,
            linear.out_features,
            rank,
            alpha,
        )
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
        # DoRA：幅度与方向分解应用于整个预训练权重
        new_weight = self.m * V

        output = F.linear(x, new_weight, self.linear.bias)

        return output


def replace_linear_with_lora(model, rank: int = 16, alpha: int = 16):
    """把模型中的所有 nn.Linear 替换为 LinearWithLoRA（递归遍历子模块）。"""
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            setattr(model, name, LinearWithLoRA(module, rank, alpha))
        else:
            replace_linear_with_lora(module, rank, alpha)


def freeze_linear_layers(model):
    """冻结所有 nn.Linear 的参数（LoRA 训练时只学旁路）。"""
    for child in model.children():
        if isinstance(child, nn.Linear):
            for param in child.parameters():
                param.requires_grad = False
        else:
            freeze_linear_layers(child)


def merge_lora_weights(module):
    """把 LinearWithLoRA 旁路权重合并回主权重并就地替换为普通 Linear。

    返回合并后的 nn.Linear（替换调用方引用）；对非 LoRA 模块原样返回。
    """
    if not isinstance(module, LinearWithLoRA):
        return module
    merged = nn.Linear(module.linear.in_features, module.linear.out_features,
                       bias=module.linear.bias is not None)
    with torch.no_grad():
        delta = module.lora.alpha * (module.lora.A @ module.lora.B).T
        merged.weight.copy_(module.linear.weight + delta)
        if module.linear.bias is not None:
            merged.bias.copy_(module.linear.bias)
    return merged.to(device=module.linear.weight.device, dtype=module.linear.weight.dtype)


# 测试代码 main 函数
def main():
    # 用法示例见 tests/unit/test_lora.py
    pass

if __name__ == "__main__":
    main()
