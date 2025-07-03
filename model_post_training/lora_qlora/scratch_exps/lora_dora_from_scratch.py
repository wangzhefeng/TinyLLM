# -*- coding: utf-8 -*-

# ***************************************************
# * File        : lora.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-07-01
# * Version     : 1.0.070123
# * Description : description
# *               https://github.com/rasbt/LLM-finetuning-scripts/blob/main/adapter/lora-from-scratch/lora-dora-mlp.ipynb
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
import copy
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn

from exp.exp_basic import Exp_Basic
from data_provider.finetune.lora.data_loader import get_dataloader
from utils.model_memory import model_memory_size
from layers.lora_dora import freeze_linear_layers, LinearWithLoRAMerged, LinearWithDoRAMerged

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]
os.environ['LOG_NAME'] = LOGGING_LABEL
from utils.log_util import logger


class Exp(Exp_Basic):

    def __init__(self, args):
        super(Exp, self).__init__(args)
    
    def _get_data(self):
        """
        数据构建
        """
        train_loader, test_loader = get_dataloader(self.args.batch_size)
        logger.info(f"\t\t\tTrain_loader steps: {len(train_loader)}, Test_loader steps: {len(test_loader)}")
        
        return train_loader, test_loader
    
    def _build_model(self):
        """
        模型构建
        """
        # model
        model = self.model_dict[self.args.model].Model(self.args)
        # multi-gpu
        if self.args.use_gpu and self.args.use_multi_gpu:
            model = nn.DataParallel(model, device_ids = self.args.devices)
        # model params size and memory size
        total_memory_size = model_memory_size(model, verbose=True)

        return model
    
    def _select_criterion(self):
        """
        选择损失函数
        """
        criterion = nn.CrossEntropyLoss()
        
        return criterion
    
    def _select_optimizer(self):
        """
        选择优化器
        """
        optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=self.args.learning_rate
        )

        return optimizer

    def train(self):
        logger.info(f"{'-' * 40}")
        logger.info("Model Train...")
        logger.info(f"{'-' * 40}")
        # dataloader
        train_loader, test_loader = self._get_data()
        # optimizer
        optimizer = self._select_optimizer()
        # criterion
        criterion = self._select_criterion()
        # model training
        train_start_time = time.time()
        for epoch in range(self.args.train_epochs):
            self.model.train()
            for batch_idx, (features, targets) in enumerate(train_loader):
                # data preprocess
                features = features.view(-1, 28 * 28).to(self.device)
                targets = targets.to(self.device)
                # FORWARD AND BACK PROP
                logits = self.model(features)
                loss = criterion(logits, targets)
                optimizer.zero_grad()
                loss.backward()
                # UPDATE MODEL PARAMETERS
                optimizer.step()
                # LOGGING
                if not batch_idx % 400:
                    logger.info(f'Epoch: {epoch+1:03d}/{self.args.train_epochs:03d} | Batch {batch_idx:03d}/{len(train_loader):03d} | Loss: {loss:.4f}')
            # training accuracy
            with torch.set_grad_enabled(False):
                train_acc = self.vali(train_loader, self.model)
                logger.info(f'Epoch: {epoch+1:03d}/{self.args.train_epochs:03d} | training accuracy: {train_acc:.2f}%')
            logger.info(f'Epoch: {epoch+1:03d}/{self.args.train_epochs:03d} | Time elapsed: {(time.time() - train_start_time)/60:.2f} min')
        logger.info(f'\t\t\t\tTotal Training Time: {(time.time() - train_start_time)/60:.2f} min')

    def vali(self, data_loader, model):
        """
        model evaluation
        """
        # model eval
        model.eval()
        # compute accuracy
        correct_pred, num_examples = 0, 0
        with torch.no_grad():
            for features, targets in data_loader:
                # data preprocess
                features = features.view(-1, 28*28).to(self.device)
                targets = targets.to(self.device)
                # forward
                logits = model(features)
                # compute accuracy
                _, predicted_labels = torch.max(logits, 1)
                num_examples += targets.size(0)
                correct_pred += (predicted_labels == targets).sum()
                acc = correct_pred.float() / num_examples * 100
            
            return acc

    def test(self):
        """
        model testing
        """
        logger.info(f"{'-' * 40}")
        logger.info("Model Testing...")
        logger.info(f"{'-' * 40}")
        # dataloader
        train_loader, test_loader = self._get_data()       
        # compute accuracy
        test_acc = self.vali(test_loader, self.model)
        logger.info(f'\t\t\t\tTest accuracy: {test_acc:.2f}%')

    def lora_finetuning(self, load: bool=False):
        # pretrained model
        if load:
            self.model.load_state_dict(torch.load(self.args.model_path))
        # model finetuning
        self.model.layers[0] = LinearWithLoRAMerged(self.model.layers[0], rank=4, alpha=8)
        self.model.layers[2] = LinearWithLoRAMerged(self.model.layers[2], rank=4, alpha=8)
        self.model.layers[4] = LinearWithLoRAMerged(self.model.layers[4], rank=4, alpha=8)
        self.model.to(self.device)
        # freeze linear layers
        if self.args.lora or self.args.dora:
            freeze_linear_layers(self.model)
            # model params size and memory size
            total_memory_size = model_memory_size(self.model, verbose=True)
        # model training and testing
        self.train()
        self.test()

    def dora_finetuning(self, load: bool=False):
        # pretrained model
        if load:
            self.model.load_state_dict(torch.load(self.args.model_path))
        # model finetuning
        self.model.layers[0] = LinearWithDoRAMerged(self.model.layers[0], rank=4, alpha=8)
        self.model.layers[2] = LinearWithDoRAMerged(self.model.layers[2], rank=4, alpha=8)
        self.model.layers[4] = LinearWithDoRAMerged(self.model.layers[4], rank=4, alpha=8)
        self.model.to(self.device)
        # freeze linear layers
        if self.args.lora or self.args.dora:
            freeze_linear_layers(self.model)
            # model params size and memory size
            total_memory_size = model_memory_size(self.model, verbose=True)
        # model training and testing
        self.train()
        self.test()




# 测试代码 main 函数
def main():
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
    # hyperparameters
    torch.manual_seed(123)

    # model and training params
    from utils.args_tools import DotDict
    args = DotDict({
        # task
        "model": "mlp",
        # model arch params
        "num_features": 784,
        "num_hidden_1": 128,
        "num_hidden_2": 256,
        "num_classes": 10,
        "lora": False,
        "dora": False,
        "rank": 4,
        "alpha": 8,
        # training params
        "batch_size": 64,
        "learning_rate": 5e-3,
        "train_epochs": 2,
        # device
        "use_gpu": 1,
        "gpu_type": "mps",
        "use_multi_gpu": 0,
        "devices": "0,1,2,3,4,5,6,7",
    })
    # ------------------------------
    # model training without LoRA
    # ------------------------------
    # exp
    exp = Exp(args)
    # model training
    exp.train()
    # model testing
    exp.test()
    # lora test
    # args.lora = True
    # exp.lora_finetuning()
    # dora test
    args.dora = True
    exp.dora_finetuning()

if __name__ == "__main__":
    main()
