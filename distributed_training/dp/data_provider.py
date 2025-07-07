# -*- coding: utf-8 -*-

# ***************************************************
# * File        : data_provider.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-07-07
# * Version     : 1.0.070714
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

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


class ToyDataset(Dataset):
    
    def __init__(self, X, y):
        self.features = X
        self.labels = y
    
    def __getitem__(self, index):
        one_x = self.features[index]
        one_y = self.labels[index]
        
        return one_x, one_y
    
    def __len__(self):
        return self.labels.shape[0]


def prepare_dataset():
    X_train = torch.tensor([
        [-1.2, 3.1],
        [-0.9, 2.9],
        [-0.5, 2.6],
        [2.3, -1.1],
        [2.7, -1.5]
    ])
    y_train = torch.tensor([0, 0, 0, 1, 1])

    X_test = torch.tensor([
        [-0.8, 2.8],
        [2.6, -1.6],
    ])
    y_test = torch.tensor([0, 1])

    # Uncomment these lines to increase the dataset size to run this script on up to 8 GPUs:
    # factor = 4
    # X_train = torch.cat([X_train + torch.randn_like(X_train) * 0.1 for _ in range(factor)])
    # y_train = y_train.repeat(factor)
    # X_test = torch.cat([X_test + torch.randn_like(X_test) * 0.1 for _ in range(factor)])
    # y_test = y_test.repeat(factor)

    train_ds = ToyDataset(X_train, y_train)
    test_ds = ToyDataset(X_test, y_test)

    train_loader = DataLoader(
        dataset=train_ds,
        batch_size=2,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )
    test_loader = DataLoader(
        dataset=test_ds,
        batch_size=2,
        shuffle=False,
    )
    
    return train_loader, test_loader




# 测试代码 main 函数
def main():
    rank=0
    world_size=1
    from distributed_training.utils.ddp_utils import ddp_setup_custom
    # initialize process group
    ddp_setup_custom(rank, world_size)
    
    train_loader, test_loader = prepare_dataset()

if __name__ == "__main__":
    main()
