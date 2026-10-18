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
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)

import numpy as np
import torch

from utils.log_util import logger

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


class EarlyStopping:
    
    def __init__(self, patience=7, verbose=False, delta=0, use_ddp=False, gpu=0):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.use_ddp = use_ddp
        self.gpu = gpu
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf

    def __call__(self, epoch, val_loss, model, optimizer=None, scheduler=None, model_path:str=""):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(epoch, val_loss, model, optimizer, scheduler, model_path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            logger.info(f"\t\t\tEpoch {epoch+1}: EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(epoch, val_loss, model, optimizer, scheduler, model_path)
            self.counter = 0

    def save_checkpoint(self, epoch, val_loss, model, optimizer, scheduler, model_path):
        # In DDP only rank 0 writes the checkpoint (other ranks have no model.module copy)
        if self.use_ddp and self.gpu != 0:
            return
        # log
        if self.verbose:
            logger.info(f"\t\tEpoch {epoch+1}: Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...")
        # checkpoint save
        ckp = model.module.state_dict() if self.use_ddp else model.state_dict()
        training_state = {
            "epoch": epoch + 1,
            "model": ckp,
            "optmizer": optimizer.state_dict() if optimizer is not None else None,
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
        }
        torch.save(training_state, model_path)
        # update minimum vali loss
        self.val_loss_min = val_loss




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
