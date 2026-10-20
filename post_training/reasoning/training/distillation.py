# -*- coding: utf-8 -*-

# ***************************************************
# * File        : distillation.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2026-10-20
# * Version     : 2.0.102021
# * Description : 知识蒸馏 from scratch（reasoning-from-scratch ch08）。
# *               教师模型（大/已训练）的软标签分布 -> 学生模型（小）：
# *               L = alpha · KL(softmax(teacher/T) || softmax(student/T)) · T²
# *                 + (1-alpha) · CE(student, hard_labels)
# *               教学实现聚焦 logit 蒸馏；on-policy 生成蒸馏可组合
# *               inference 采样 + 本模块的 loss。
# * Link        : Distilling the Knowledge in a Neural Network
# *               https://arxiv.org/abs/1503.02531
# * Requirement : torch
# ***************************************************

__all__ = [
    "distillation_loss",
    "distill_step",
]

# python libraries
from pathlib import Path

import torch
import torch.nn.functional as F

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


def distillation_loss(student_logits, teacher_logits, temperature=2.0,
                      alpha=1.0, hard_labels=None):
    """KL 蒸馏损失（Hinton 2015）。

    Args:
        student_logits: (B, T, V) 学生模型输出
        teacher_logits: (B, T, V) 教师模型输出（应 detached）
        temperature: 温度 T；越高分布越平滑，软标签信息越丰富
        alpha: 软损失权重；1.0 = 纯蒸馏，0.0 = 纯硬标签 CE
        hard_labels: (B, T) 真实标签（alpha < 1 时必填）

    Returns:
        标量 loss（可反传，仅对学生参数）
    """
    t = teacher_logits / temperature
    s = student_logits / temperature
    log_p_teacher = F.log_softmax(t, dim=-1)
    log_p_student = F.log_softmax(s, dim=-1)
    # KL(teacher || student) = Σ p_t · (log p_t - log p_s)；
    # kl_div(input=log_p_student, target=p_teacher, log_target=False) 即该式
    kd = F.kl_div(log_p_student, log_p_teacher.exp(), reduction="batchmean") * (temperature ** 2)

    if alpha >= 1.0:
        return kd
    if hard_labels is None:
        raise ValueError("alpha < 1.0 时必须提供 hard_labels")
    ce = F.cross_entropy(
        student_logits.reshape(-1, student_logits.size(-1)),
        hard_labels.reshape(-1),
    )
    return alpha * kd + (1.0 - alpha) * ce


def distill_step(student, teacher, input_ids, optimizer, temperature=2.0,
                 alpha=1.0, hard_labels=None):
    """单步蒸馏：教师前向（无梯度）-> 学生前向 -> 蒸馏 loss -> 反传更新。

    Args:
        student: 可训练学生模型，forward(x) -> (B, T, V)
        teacher: 冻结教师模型（eval 模式，结构与词表兼容）
        input_ids: (B, T)
        hard_labels: alpha < 1 时的硬标签（通常为 input_ids 左移一位）

    Returns:
        dict(loss=..., kd=..., ce=...)
    """
    student.train()
    teacher.eval()
    with torch.no_grad():
        teacher_logits = teacher(input_ids)

    optimizer.zero_grad()
    student_logits = student(input_ids)
    loss = distillation_loss(
        student_logits, teacher_logits,
        temperature=temperature, alpha=alpha, hard_labels=hard_labels,
    )
    loss.backward()
    optimizer.step()
    return {"loss": float(loss.item())}


# 测试代码 main 函数
def main():
    # 用法示例见 tests/unit/test_distillation.py
    pass

if __name__ == "__main__":
    main()
