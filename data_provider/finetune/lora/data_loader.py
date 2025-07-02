# -*- coding: utf-8 -*-

# ***************************************************
# * File        : data_loader.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-07-02
# * Version     : 1.0.070210
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
import warnings
warnings.filterwarnings("ignore")

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]

from utils.log_util import logger


def get_dataloader(batch_size):
    """
    MINIST dataset
    """
    # Note transforms.ToTensor() scales input images
    # to 0-1 range
    train_dataset = datasets.MNIST(
        root='data', 
        train=True, 
        transform=transforms.ToTensor(),
        download=True
    )
    test_dataset = datasets.MNIST(
        root='data', 
        train=False, 
        transform=transforms.ToTensor()
    )
    train_loader = DataLoader(
        dataset=train_dataset, 
        batch_size=batch_size, 
        shuffle=True
    )
    test_loader = DataLoader(
        dataset=test_dataset, 
        batch_size=batch_size, 
        shuffle=False
    )
    # Checking the dataset
    # for images, labels in train_loader:
    #     logger.info(f"Image batch dimensions: {images.shape}")
    #     logger.info(f"Image label dimensions: {labels.shape}")
    #     break

    return train_loader, test_loader




# 测试代码 main 函数
def main():
    train_loader, test_loader = get_dataloader(batch_size=64)

if __name__ == "__main__":
    main()
