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


# TODOs
def learning_rate_warmup(optimizer, 
                         train_loader, 
                         train_epochs, 
                         global_step, 
                         initial_lr = 3e-5, 
                         min_lr=1e-6):
    """
    learning rate warmup

    Args:
        optimizer (_type_): _description_
        train_loader (_type_): _description_
        train_epochs (_type_): _description_
        global_step (_type_): _description_
        initial_lr (_type_, optional): _description_. Defaults to 3e-5.
        min_lr (_type_, optional): _description_. Defaults to 1e-6.

    Returns:
        _type_: _description_
    """
    track_lrs = []
    # 从优化器中获取最大学习率
    peak_lr = optimizer.param_groups[0]["lr"]
    peak_lr = 0.001
    # 计算训练过程中总的迭代次数
    total_training_steps = len(train_loader) * train_epochs
    # warmup steps
    warmup_steps = int(0.2 * total_training_steps)
    # 计算warmup阶段的迭代次数
    lr_increment = (peak_lr - initial_lr) / warmup_steps
    
    # 根据当前阶段（预热或余弦衰减）调整学习率
    if global_step < warmup_steps:
        # 线性预热
        lr = initial_lr + global_step * lr_increment
    else:
        # 预热后余弦衰减
        progress = (global_step - warmup_steps) / (total_training_steps - warmup_steps)
        lr = min_lr + (peak_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
    
    # 将计算出的学习率应用到优化器中
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    # 记录当前学习率
    track_lrs.append(lr)
    
    return track_lrs





# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
