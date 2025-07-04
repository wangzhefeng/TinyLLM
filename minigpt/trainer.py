# -*- coding: utf-8 -*-

# ***************************************************
# * File        : trainer.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-07-04
# * Version     : 1.0.070413
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
import time
import platform
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")

import torch
from torch.utils.data import DataLoader, RandomSampler

from minigpt.utils import CfgNode as CN
from utils.log_util import logger

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


class Trainer:
    
    @staticmethod
    def get_default_config():
        C = CN()
        # device to train on
        C.device = "auto"
        # dataloader parameters
        C.num_workers = 4 if platform.system() != "Windows" else 0
        # optimizer parameters
        C.max_iters = None
        C.batch_size = 64
        C.learning_rate = 3e-4
        C.betas = (0.9, 0.95)
        C.weight_decay = 0.1  # only applied on matmul weights
        C.grad_norm_clip = 1.0

        return C

    def __init__(self, config, model, train_dataset):
        self.config = config
        self.model = model
        self.optimizer = None
        self.train_dataset = train_dataset
        self.callbacks = defaultdict(list)
        
        # determine the device we'll train on
        if config.device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = config.device
        # move the model to the device
        self.model = self.model.to(self.device)
        logger.info(f"running on device: {self.device}")
        
        # variables that will be assigned to trainer class later for logging and etc
        self.iter_num = 0  # training iteration number
        self.iter_time = 0.0  # training timestamp
        self.iter_dt = 0.0  # training iteration time

    def add_callback(self, onevent: str, callback):
        self.callbacks[onevent].append(callback)

    def set_callback(self, onevent: str, callback):
        self.callbacks[onevent] = [callback]

    def trigger_callback(self, onevent: str):
        for callback in self.callbacks.get(onevent, []):
            callback(self)

    def run(self):
        # config
        config = self.config
        # model
        model = self.model
        # setup the optmizer
        self.optimizer = model.configure_optimizers(config)
        # setup the dataloader
        train_loader = DataLoader(
            self.train_dataset, 
            batch_size=config.batch_size, 
            shuffle=False,
            sampler=RandomSampler(
                self.train_dataset, 
                replacement=True,
                num_samples=int(1e10)
            ),
            pin_memory=True,
            num_workers=config.num_workers
        )
        # training state params
        model.train()
        self.iter_num = 0
        self.iter_time = time.time()
        # training loop
        data_iter = iter(train_loader)
        while True:
            # fetch the next batch (x, y) and re-init iterator if needed
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)
            batch = [t.to(self.device) for t in batch]
            x, y = batch
            
            # forward the model
            logits, self.loss = model(x, y)

            # backprop and update the parameters
            model.zero_grad(set_to_none=True)
            self.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
            self.optimizer.step()

            # update training state params
            self.trigger_callback("on_batch_end")
            self.iter_num += 1
            tnow = time.time()
            self.iter_dt = tnow - self.iter_time
            self.iter_time = tnow
            
            # termination conditions
            if config.max_iters is not None and self.iter_num >= config.max_iters:
                break




# 测试代码 main 函数
def main():
    config = None
    model = None
    train_dataset = None
    trainer = Trainer(config, model, train_dataset)
    trainer.run()

if __name__ == "__main__":
    main()
