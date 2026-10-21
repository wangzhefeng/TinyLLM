# -*- coding: utf-8 -*-

# ***************************************************
# * File        : calc_accuracy.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2026-10-21
# * Version     : 1.0.102121
# * Description : 分类微调的准确率计算（LLMs-from-scratch ch06）。
# *               与 utils/calc_loss.py 的 CLF_TASK_NAMES 分支保持同一约定：
# *               分类任务取最后一个 token 的 logits 做 argmax。
# * Link        : https://github.com/rasbt/LLMs-from-scratch/tree/main/ch06
# * Requirement : torch
# ***************************************************

__all__ = [
    "calc_accuracy_loader",
    "calc_final_accuracy",
]

# python libraries
from pathlib import Path

import torch

from utils.calc_loss import CLF_TASK_NAMES
from utils.log_util import logger

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


def calc_accuracy_loader(data_loader, model, device, num_batches=None, task_name="clf_sft"):
    """
    计算 num_batches 个批次的分类准确率（LLMs-from-scratch ch06 的 calc_accuracy_loader）。

    Args:
        data_loader: 每个批次 (input_batch, target_batch)；target 为类别索引
        model: 分类微调模型（out_head 输出 num_classes 维 logits）
        device: torch device
        num_batches: None 时遍历整个 loader
        task_name: 分类任务名（决定是否取最后一个 token 的 logits）

    Returns:
        accuracy: float，正确数 / 样本数
    """
    # eval mode
    model.eval()
    # 准确预测的数量、样本总数
    correct_predictions, num_examples = 0, 0
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    with torch.no_grad():
        for i, (input_batch, target_batch) in enumerate(data_loader):
            if i >= num_batches:
                break
            input_batch, target_batch = input_batch.to(device), target_batch.to(device)
            if task_name in CLF_TASK_NAMES:
                # 最后一个 token 的 logits -> 类别
                logits = model(input_batch)[:, -1, :]
            else:
                logits = model(input_batch)
            predicted_labels = torch.argmax(logits, dim=-1)
            num_examples += predicted_labels.shape[0]
            correct_predictions += sum(predicted_labels == target_batch).item()
    # train mode 恢复交给调用方；这里保持 eval（与书一致由调用方切换）
    return correct_predictions / num_examples


def calc_final_accuracy(train_loader, valid_loader, test_loader, model, device, num_batches=None, task_name="clf_sft"):
    """
    训练结束后在整个 train/valid/test 上计算并打印最终准确率
    （对应书中 ch06 末尾的 full-dataset accuracy 统计）。

    Returns:
        dict(train=..., valid=..., test=...)
    """
    model.eval()
    res = {}
    for name, loader in (("train", train_loader), ("valid", valid_loader), ("test", test_loader)):
        acc = calc_accuracy_loader(loader, model, device, num_batches=num_batches, task_name=task_name)
        res[name] = acc
        logger.info(f"{name} accuracy: {acc * 100:.2f}%")
    model.train()

    return res


# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
