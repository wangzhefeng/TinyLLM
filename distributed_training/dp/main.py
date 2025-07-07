# -*- coding: utf-8 -*-

# ***************************************************
# * File        : main_torchrun.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-07-07
# * Version     : 1.0.070714
# * Description : description
# * Link        : https://docs.pytorch.org/tutorials/beginner/blitz/data_parallel_tutorial.html
# *               https://www.cnblogs.com/picassooo/p/16473072.html
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

import torch
import torch.nn as nn
import torch.nn.functional as F

from distributed_training.dp.data_provider import prepare_dataset
from distributed_training.dp.model import NeuralNetwork

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]
os.environ['LOG_NAME'] = LOGGING_LABEL
from utils.log_util import logger


def compute_accuracy(model, dataloader, device):
    model = model.eval()
    correct = 0.0
    total_examples = 0

    for idx, (features, labels) in enumerate(dataloader):
        features, labels = features.to(device), labels.to(device)

        with torch.no_grad():
            logits = model(features)
        predictions = torch.argmax(logits, dim=1)
        compare = labels == predictions
        correct += torch.sum(compare)
        total_examples += len(compare)
    return (correct / total_examples).item()


def train(device, num_epochs):
    # data prepare
    train_loader, test_loader = prepare_dataset()
    # model
    model = NeuralNetwork(num_inputs=2, num_outputs=2)
    # optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
    
    # wrap model with DP(the core model is now accessible as model.module)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(
            model, 
            # device_ids=[range(world_size)], 
            # output_device=rank
        )
    # copy model to multi-GPUs
    model.to(device)

    for epoch in range(num_epochs):
        # training mode
        model.train()
        for features, labels in train_loader:
            # use rank
            features, labels = features.to(device), labels.to(device)
            # forward
            logits = model(features)
            loss = F.cross_entropy(logits, labels)
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # log
            logger.info(f"[GPU{str(device)[-1]}] Epoch: {epoch+1:03d}/{num_epochs:03d}"
                  f" | Batchsize {labels.shape[0]:03d}"
                  f" | Train/Val Loss: {loss:.2f}")
    model.eval()

    try:
        train_acc = compute_accuracy(model, train_loader, device=device)
        logger.info(f"[GPU{str(device)[-1]}] Training accuracy: {train_acc:.4f}")
        test_acc = compute_accuracy(model, test_loader, device=device)
        logger.info(f"[GPU{str(device)[-1]}] Test accuracy: {test_acc:.4f}")
    except ZeroDivisionError as e:
        pass


def save_model(model, checkpoint: str):
    torch.save(model.module.cpu().state_dict(), checkpoint)


def load_model(model, checkpoint: str):
    model.load_state_dict(torch.load(checkpoint))
    return model




# 测试代码 main 函数
def main():
    # set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    logger.info(f"Number of GPUs available: {torch.cuda.device_count()}")
    logger.info(f"Use device: {device}")
    
    # set seed
    torch.manual_seed(123)

    # training
    train(device, num_epochs=3)

if __name__ == "__main__":
    main()
