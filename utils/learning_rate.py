# -*- coding: utf-8 -*-

# ***************************************************
# * File        : learning_rate.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-09-13
# * Version     : 1.0.091321
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

__all__ = []

# python libraries
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)
import math
import warnings
warnings.filterwarnings("ignore")

from utils.log_util import logger

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


def adjust_learning_rate(optimizer, epoch, args):
    """
    学习率调整

    Args:
        optimizer (_type_): _description_
        epoch (_type_): _description_
        args (_type_): _description_
    """
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    lr_adjust = {}
    if args.lradj == 'type1':
        lr_adjust = {
            epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))
        }
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 
            4: 1e-5, 
            6: 5e-6, 
            8: 1e-6, 
            10: 5e-7, 
            15: 1e-7, 
            20: 5e-8,
        }
    elif args.lradj == '3':
        lr_adjust = {
            epoch: args.learning_rate if epoch < 10 else args.learning_rate*0.1
        }
    elif args.lradj == '4':
        lr_adjust = {
            epoch: args.learning_rate if epoch < 15 else args.learning_rate*0.1
        }
    elif args.lradj == '5':
        lr_adjust = {
            epoch: args.learning_rate if epoch < 25 else args.learning_rate*0.1
        }
    elif args.lradj == '6':
        lr_adjust = {
            epoch: args.learning_rate if epoch < 5 else args.learning_rate*0.1
        } 
    
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        logger.info(f'\tEpoch {epoch}: Updating learning rate to {lr}')


def warmup_cosine_lr(global_step: int,
                     total_steps: int,
                     warmup_steps: int,
                     peak_lr: float,
                     initial_lr: float = 3e-5,
                     min_lr: float = 1e-6) -> float:
    """
    step 级学习率：线性 warmup + 余弦衰减（LLMs-from-scratch 附录 D）。

    Args:
        global_step: 当前 step（从 0 开始）
        total_steps: 总训练 step 数
        warmup_steps: warmup step 数
        peak_lr: 峰值学习率（即 optimizer 的学习率）
        initial_lr: warmup 起始学习率
        min_lr: 余弦衰减的最低学习率

    Returns:
        当前 step 的学习率
    """
    if total_steps <= 0:
        raise ValueError(f"total_steps must be positive, got {total_steps}")
    warmup_steps = max(0, min(warmup_steps, total_steps - 1))
    # 线性预热
    if global_step < warmup_steps:
        lr_increment = (peak_lr - initial_lr) / max(warmup_steps, 1)
        return initial_lr + global_step * lr_increment
    # 预热后余弦衰减
    decay_steps = max(total_steps - warmup_steps, 1)
    progress = min((global_step - warmup_steps) / decay_steps, 1.0)
    return min_lr + (peak_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))


def set_optimizer_lr(optimizer, lr: float) -> float:
    """
    把学习率写入 optimizer 的所有 param group，返回写入的值。
    """
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


def learning_rate_warmup(optimizer,
                         train_loader,
                         train_epochs,
                         global_step,
                         initial_lr = 3e-5,
                         min_lr=1e-6):
    """
    learning rate warmup（已迁移到 warmup_cosine_lr + set_optimizer_lr，保留兼容）

    Returns:
        list: 当前 lr 组成的列表（历史行为）
    """
    # 从优化器中获取最大学习率
    peak_lr = optimizer.param_groups[0]["lr"]
    # 计算训练过程中总的迭代次数
    total_training_steps = len(train_loader) * train_epochs
    # warmup steps
    warmup_steps = int(0.2 * total_training_steps)
    # 根据当前阶段（预热或余弦衰减）计算学习率
    lr = warmup_cosine_lr(
        global_step=global_step,
        total_steps=total_training_steps,
        warmup_steps=warmup_steps,
        peak_lr=peak_lr,
        initial_lr=initial_lr,
        min_lr=min_lr,
    )
    # 将计算出的学习率应用到优化器中
    set_optimizer_lr(optimizer, lr)

    return [lr]





# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
