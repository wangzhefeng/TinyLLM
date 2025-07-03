# -*- coding: utf-8 -*-

# ***************************************************
# * File        : train_funcs.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2025-02-23
# * Version     : 0.1.022300
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


import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from utils.log_util import logger

plt.switch_backend('agg')

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


def select_optimizer(model, learning_rate: float, weight_decay: float):
    """
    optimizer
    """
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr = learning_rate, 
        weight_decay = weight_decay
    )

    return optimizer


def select_criterion():
    """
    loss
    """
    criterion = nn.CrossEntropyLoss()

    return criterion


def adjust_learning_rate(optimizer, epoch, args):
    """
    学习率调整

    Args:
        optimizer (_type_): _description_
        epoch (_type_): _description_
        args (_type_): _description_
    """
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
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


# TODO
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


# TODO
def gradient_clipping(model, global_step, warmup_steps):
    """
    在预热阶段后应用梯度裁剪，防止梯度爆炸
    """
    if global_step >= warmup_steps:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 1.0)

    return model


class EarlyStopping:
    
    def __init__(self, patience=7, verbose=False, delta=0, use_ddp=False, gpu_id=0):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.use_ddp = use_ddp
        self.gpu_id = self.gpu_id
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf

    def __call__(self, epoch, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(epoch, val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            logger.info(f"\t\t\tEpoch {epoch}: EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(epoch, val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, epoch, val_loss, model, path):
        if self.verbose:
            logger.info(f"\t\tEpoch {epoch}: Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...")
        # checkpoint save
        if not self.use_ddp:
            ckp = model.state_dict()
            torch.save(ckp, path)
        elif self.use_ddp and self.gpu == 0:
            ckp = model.module.state_dict()
            torch.save(ckp, path)
        # update minimum vali loss
        self.val_loss_min = val_loss




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
