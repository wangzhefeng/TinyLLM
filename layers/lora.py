# -*- coding: utf-8 -*-

# ***************************************************
# * File        : lora.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2026-10-21
# * Version     : 1.1.102121
# * Description : LoRA / DoRA 参数高效微调层（LLMs-from-scratch 附录 E）。
# *               规范实现放在 layers/（供任意 models/ 模型即插即用）；
# *               （原 post_training/peft/layers/lora_dora.py 兼容 shim 已移除）
# *               v1.1 变更（对照附录 E 最新版 notebook）：
# *               1. 缩放改为标准 alpha/rank（LoRA 论文与 HF peft 惯例，
# *                  换 rank 无需重调学习率）
# *               2. LoRA 分支支持 dropout
# *               3. replace_linear_with_lora 支持 target/exclude 关键词过滤
# *               4. DoRA 简化版两变体的不等价性显式文档化 + 数值保护统一
# *               5. 新增论文版 LinearWithDoRACorrect（对 W+ΔW 做 m·V 分解）
# *               6. merge_lora_weights 对 DoRA 变体显式 raise TypeError
# * Link        : LoRA https://arxiv.org/abs/2106.09685
# *               DoRA https://arxiv.org/abs/2402.09353
# *               LLMs-from-scratch 附录 E
# *               https://github.com/rasbt/LLMs-from-scratch/tree/main/appendix-E
# * Requirement : torch
# ***************************************************

__all__ = [
    "LoRALayer",
    "LinearWithLoRA",
    "LinearWithDoRA",
    "LinearWithDoRACorrect",
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
    """低秩适配 ΔW = (alpha / rank) · (A @ B)，A: (in, rank)，B: (rank, out)。

    - rank(r)：低秩内维，控制新增参数量（参数效率 vs 适应能力的平衡）
    - alpha：缩放超参。实际缩放为 **alpha / rank**（LoRA 论式 5 与 HF peft
      的 lora_alpha/r 惯例；LLMs-from-scratch 附录 E 最新版 notebook 已改为
      同一公式），好处是更换 rank 时无需重新调学习率
    - dropout：只作用于 LoRA 分支输入的 dropout（默认 0.0 = 关闭，
      保持本仓库旧行为）
    - B 初始化为零 → 初始 ΔW = 0，不破坏预训练行为

    注：附录 E 最新版将 A 的初始化改为 kaiming_uniform_(a=sqrt(5))
    （与 nn.Linear 默认一致）；本实现保留原先的 1/sqrt(r) 高斯初始化，
    两者对 B=0 的初始行为等价。
    """

    def __init__(self, in_dim, out_dim, rank, alpha, dropout: float = 0.0):
        super().__init__()

        std_dev = 1 / torch.sqrt(torch.tensor(rank).float())
        self.A = nn.Parameter(torch.randn(in_dim, rank) * std_dev)
        self.B = nn.Parameter(torch.zeros(rank, out_dim))   # B 初始为零：ΔW 从零开始
        self.alpha = alpha
        self.rank = rank
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(x)
        return (self.alpha / self.rank) * (x @ self.A @ self.B)


class LinearWithLoRA(nn.Module):
    """线性层 + LoRA 旁路（不合并版）：y = Wx + (alpha/r)·xAB。"""

    def __init__(self, linear, rank, alpha, dropout: float = 0.0):
        super().__init__()

        self.linear = linear
        self.lora = LoRALayer(
            linear.in_features,
            linear.out_features,
            rank,
            alpha,
            dropout,
        )

    def forward(self, x):
        return self.linear(x) + self.lora(x)


class LinearWithDoRA(nn.Module):
    """线性层 + DoRA（不合并版，**简化实现**）。

    前向：y = Wx + m ⊙ normalize(LoRA_branch(x))

    - 只对 **LoRA 旁路输出** 按行（batch 维）做 L2 归一化，再乘可学习幅度
      m（shape (1, out_features)），基座输出 Wx 不参与归一化
    - 这是 LLMs-from-scratch 附录 E 简化版的忠实移植；**原论文 DoRA**
      (https://arxiv.org/abs/2402.09353, Alg. 1) 是对组合权重 W+ΔW 做
      幅度-方向分解（见 LinearWithDoRACorrect）
    - 注意：本类与 LinearWithDoRAMerged **数学不等价**（后者做权重分解），
      即 merge 前后行为不同——这是书简化版的已知特性，教学上可对比讨论
    """

    def __init__(self, linear, rank, alpha, dropout: float = 0.0):
        super().__init__()

        self.linear = linear
        self.lora = LoRALayer(
            linear.in_features,
            linear.out_features,
            rank,
            alpha,
            dropout,
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


class LinearWithDoRACorrect(nn.Module):
    """DoRA 论文版（Liu et al. 2024）：对 **组合权重** W + ΔW 做幅度-方向分解。

    前向：W' = m ⊙ V，V = (W + (alpha/r)·ΔW) / ||W + (alpha/r)·ΔW||_row

    - 每个输出神经元的权重向量是 weight（out, in）的**一行**，因此范数沿
      in_features 维（dim=1）计算，m ∈ R^{out}（论文记号 m ∈ R^{d_out}）
    - m 可学习，初始化为原权重的行范数；B=0 时 W' = W，初始不改变行为
    - 与附录 E 简化版（LinearWithDoRA / LinearWithDoRAMerged）不同，本类
      是论文语义；合并版可在推断时直接用本前向（本身就是权重级计算）
    """

    def __init__(self, linear, rank, alpha, dropout: float = 0.0):
        super().__init__()

        self.linear = linear
        self.lora = LoRALayer(
            linear.in_features,
            linear.out_features,
            rank,
            alpha,
            dropout,
        )
        # (out,)：每个输出神经元一个幅度
        self.m = nn.Parameter(linear.weight.norm(p=2, dim=1))

    def forward(self, x):
        lora = self.lora.A @ self.lora.B                        # (in, out)
        combined_weight = self.linear.weight + (self.lora.alpha / self.lora.rank) * lora.T
        v = combined_weight / (combined_weight.norm(p=2, dim=1, keepdim=True) + 1e-9)
        new_weight = self.m.unsqueeze(1) * v                    # (out, in)

        return F.linear(x, new_weight, self.linear.bias)


class LinearWithLoRAMerged(nn.Module):
    """线性层 + LoRA（合并版）：推断时把 (alpha/r)·BA^T 加进权重，无旁路开销。"""

    def __init__(self, linear, rank, alpha, dropout: float = 0.0) -> None:
        super().__init__()

        self.linear = linear
        self.lora = LoRALayer(
            linear.in_features,
            linear.out_features,
            rank,
            alpha,
            dropout,
        )

    def forward(self, x):
        lora = self.lora.A @ self.lora.B
        combined_weight = self.linear.weight + (self.lora.alpha / self.lora.rank) * lora.T

        return F.linear(x, combined_weight, self.linear.bias)


class LinearWithDoRAMerged(nn.Module):
    """线性层 + DoRA（合并版，**简化实现**）：权重分解为幅度 m × 方向 V。

    前向：W' = m ⊙ V，V = (W + (alpha/r)·ΔW) / column_norm

    - 范数沿 dim=0 计算（附录 E 原实现），即 m 是**每个输入维度**一个幅度
      （shape (1, in_features)）；与论文的"每个输出神经元一个幅度"不同
    - **与 LinearWithDoRA（不合并版）数学不等价**：后者只归一化旁路激活。
      二者均来自书附录 E 的简化版，merge 前后行为不同是已知特性；
      论文语义请用 LinearWithDoRACorrect
    """

    def __init__(self, linear, rank, alpha, dropout: float = 0.0) -> None:
        super().__init__()

        self.linear = linear
        self.lora = LoRALayer(
            linear.in_features,
            linear.out_features,
            rank,
            alpha,
            dropout,
        )
        self.m = nn.Parameter(
            self.linear.weight.norm(p=2, dim=0, keepdim=True)
        )

    def forward(self, x):
        # combine LoRA matrices
        lora = self.lora.A @ self.lora.B
        # combine LoRA with original weights
        combined_weight = self.linear.weight + (self.lora.alpha / self.lora.rank) * lora.T
        # directional matrix（单位方向）
        column_norm = combined_weight.norm(p=2, dim=0, keepdim=True)
        V = combined_weight / (column_norm + 1e-9)
        # DoRA：幅度与方向分解应用于整个组合权重
        new_weight = self.m * V

        output = F.linear(x, new_weight, self.linear.bias)

        return output


def replace_linear_with_lora(model, rank: int = 16, alpha: int = 16,
                             dropout: float = 0.0,
                             target_keywords=None, exclude_keywords=None):
    """递归把模型中的 nn.Linear 替换为 LinearWithLoRA。

    Args:
        model: 任意 nn.Module（如 models/gpt2 的 GPT 模型）
        rank / alpha / dropout: LoRA 超参（缩放为 alpha/rank）
        target_keywords: None 时替换**全部** nn.Linear（旧行为）；
            否则只替换模块名（named_children 的属性名）中包含任一关键词的
            nn.Linear。典型用法：只给注意力投影加 LoRA ——
            ``target_keywords=["W_query", "W_key", "W_value"]``
        exclude_keywords: 排除模块名中包含任一关键词的层（优先于 target）。
            典型用法：分类微调时不给新建分类头加 LoRA ——
            ``exclude_keywords=["out_head"]``

    Returns:
        model（就地修改，返回便于链式调用）
    """
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            hit = target_keywords is None or any(k in name for k in target_keywords)
            excluded = exclude_keywords is not None and any(k in name for k in exclude_keywords)
            if hit and not excluded:
                setattr(model, name, LinearWithLoRA(module, rank, alpha, dropout))
        else:
            replace_linear_with_lora(module, rank, alpha, dropout,
                                     target_keywords, exclude_keywords)
    return model


def freeze_linear_layers(model):
    """冻结所有 nn.Linear 的参数（LoRA 训练时只学旁路）。

    注意：只冻结 nn.Linear（包括被 LinearWith* 包裹前的原 linear 主权重）；
    embedding / norm 保持可训练。
    """
    for child in model.children():
        if isinstance(child, nn.Linear):
            for param in child.parameters():
                param.requires_grad = False
        else:
            freeze_linear_layers(child)


def _to_merged_linear(linear: nn.Linear, lora: LoRALayer) -> nn.Linear:
    """按 alpha/rank 缩放把 LoRA 旁路折叠进主权重，返回新的普通 Linear。"""
    merged = nn.Linear(linear.in_features, linear.out_features,
                       bias=linear.bias is not None)
    with torch.no_grad():
        delta = (lora.alpha / lora.rank) * (lora.A @ lora.B).T
        merged.weight.copy_(linear.weight + delta)
        if linear.bias is not None:
            merged.bias.copy_(linear.bias)
    return merged.to(device=linear.weight.device, dtype=linear.weight.dtype)


def merge_lora_weights(module):
    """把 LoRA 旁路权重合并回主权重，返回**普通 nn.Linear**。

    - LinearWithLoRA / LinearWithLoRAMerged：返回折叠了 (alpha/r)·AB 的新 Linear
    - 普通 nn.Linear：原样返回（本就没有旁路可合并）
    - **DoRA 变体（LinearWithDoRA / LinearWithDoRAMerged / LinearWithDoRACorrect）：
      raise TypeError**。DoRA 的合并是 m·V 权重重建而非简单加 ΔW，静默返回
      原模块会让调用方误以为已合并（残留可训练的 lora/m 参数）；
      如确需合并请参考 DoRA 论文 Alg. 1 手工重建 W' = m·V。
    """
    if isinstance(module, nn.Linear):
        return module
    if isinstance(module, (LinearWithLoRA, LinearWithLoRAMerged)):
        return _to_merged_linear(module.linear, module.lora)
    raise TypeError(
        f"merge_lora_weights 不支持 {type(module).__name__}："
        "DoRA 的合并是 m·V 权重重建而非简单加 ΔW，请保留模块原样推断，"
        "或参考 https://arxiv.org/abs/2402.09353 Alg. 1 手工合并"
    )




# 测试代码 main 函数
def main():
    # 用法示例见 tests/unit/test_lora.py
    pass

if __name__ == "__main__":
    main()
