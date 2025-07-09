# -*- coding: utf-8 -*-

# ***************************************************
# * File        : model_tools.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-01-13
# * Version     : 1.0.011322
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

__all__ = []

# python libraries
import os
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)
import math


import numpy as np
import matplotlib.pyplot as plt
import torch

from utils.log_util import logger

plt.switch_backend('agg')

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
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == 'type3':
        lr_adjust = {epoch: args.learning_rate if epoch < 3 else args.learning_rate * (0.9 ** ((epoch - 3) // 1))}
    elif args.lradj == '3':
        lr_adjust = {epoch: args.learning_rate if epoch < 10 else args.learning_rate*0.1}
    elif args.lradj == '4':
        lr_adjust = {epoch: args.learning_rate if epoch < 15 else args.learning_rate*0.1}
    elif args.lradj == '5':
        lr_adjust = {epoch: args.learning_rate if epoch < 25 else args.learning_rate*0.1}
    elif args.lradj == '6':
        lr_adjust = {epoch: args.learning_rate if epoch < 5 else args.learning_rate*0.1}  
    
    elif args.lradj == "cosine":
        lr_adjust = {epoch: args.learning_rate /2 * (1 + math.cos(epoch / args.train_epochs * math.pi))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        logger.info(f'Epoch: {epoch}, \tUpdating learning rate to {lr}')


class EarlyStopping:
    
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta

    def __call__(self, val_loss, epoch, model, optimizer=None, scheduler=None, model_path=""):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, epoch, model, optimizer, scheduler, model_path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            logger.info(f'Epoch: {epoch+1}, \tEarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, epoch, model, optimizer, scheduler, model_path)
            self.counter = 0

    def save_checkpoint(self, val_loss, epoch, model, optimizer=None, scheduler=None, model_path: str=""):
        # 日志打印
        if self.verbose:
            logger.info(f'Epoch: {epoch+1}, \tValidation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')
        # 模型保存
        training_state = {
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "optmizer": optimizer.state_dict() if optimizer is not None else None,
            # "scheduler": scheduler.state_dict() if scheduler is not None else None,
        }
        torch.save(training_state, model_path)
        self.val_loss_min = val_loss


# class EarlyStopping_paddle_version:
#     """
#     早停机制
#     checkpoint 保存
#     """

#     def __init__(self, patience = 7, verbose = False, delta = 0):
#         self.patience = patience  # val_loss 不更新容忍 epoch 次数，超过 patience，则停止训练
#         self.verbose = verbose  # 是否输出日志信息
#         self.delta = delta  # 阈值

#         self.best_score = None  # 最佳评估分数
#         self.counter = 0  # 当前 epoch 的 val_loss 没有超过历史最佳分数的计数器
#         self.early_stop = False  # 是否早停
#         self.val_loss_min = np.Inf  # 最小 val_loss

#     def __call__(self, val_loss, model, path):
#         score = -val_loss  # loss 递增
#         # 首轮，直接更新 best_score 和保存节点模型
#         if self.best_score is None:
#             self.best_score = score
#             self.save_checkpoint(val_loss, model, path)
#         # 若当前 epoch 表现没超过历史最佳分数，且累积发生次数超过 patience，早停
#         elif score < self.best_score + self.delta:
#             self.counter += 1
#             print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
#             if self.counter >= self.patience:
#                 self.early_stop = True
#         # 若当前 epoch 表现超过历史最佳分数，更新 best_socre，保存该节点模型
#         else:
#             self.best_score = score
#             self.save_checkpoint(val_loss, model, path)
#             self.counter = 0

#     def save_checkpoint(self, val_loss, model, path):
#         """
#         checkpoint 保存

#         Args:
#             val_loss (_type_): 验证损失
#             model (_type_): 模型
#             path (_type_): checkpoint 保存路径
#         """
#         # 验证损失变化日志打印
#         if self.verbose:
#             print(f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...")
#         # checkpoint 保存
#         paddle.save(model.state_dict(), Path(path).joinpath("checkpoint.pth"))
#         # 更新最小验证损失
#         self.val_loss_min = val_loss


def adjustment(gt, pred):
    """
    异常检测

    Args:
        gt (_type_): _description_
        pred (_type_): _description_

    Returns:
        _type_: _description_
    """
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
            
    return gt, pred




# 测试代码 main 函数
def main():
   class Config:
       train_epochs = 10
       learning_rate = 1e-3
       lradj = 'type1'

   config = Config()
   
   adjust_learning_rate(optimizer=None, epoch=1, args=config)
       
if __name__ == "__main__":
   main()
