# -*- coding: utf-8 -*-

# ***************************************************
# * File        : distributed_sampler_training.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-01-02
# * Version     : 1.0.010222
# * Description : 使用 PyTorch DistributedDataParallel 实现单个服务器多加速卡训练
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# * TODO        : 1.
# ***************************************************

__all__ = []

# python libraries
import os
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)
import argparse
import warnings

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from models import DeepLab
from dataset import Cityscaples

warnings.filterwarnings("ignore")

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


# 参数设置
parser = argparse.ArgumentParser(description="DeepLab")
parser.add_argument("-j", "--workers", default=4, type=int, metavar="N",
                    help="number of data loading workers (default: 4)")
parser.add_argument("--epochs", default=200, type=int, metavar="N",
                    help="number of total epochs to run (default: 200)")
parser.add_argument("--start-epoch", default=0, type=int, metavar="N",
                    help="manual epoch number (useful on restarts) (default: 0)")
parser.add_argument("-b", "--batch-size", default=3, type=int, metavar="N", 
                    help="mini-batch size (default: 4)")
parser.add_argument("--local_rank", default=-0, type=int, 
                    help="node rank for distributed training")
args = parser.parse_args()


# 初始化分布式训练环境
torch.distributed.init_process_group(backend="nccl")
print(f"Use GPU: {args.local_rank}")


# 创建模型
model = DeepLab()

# 当前显卡
torch.cuda.set_device(args.local_rank)

# 模型加载到当前显卡
model = model.cuda()

# 数据并行
model = torch.nn.parallel.DistributedDataParallel(
    model,
    device_ids=[args.local_rank],
    output_device=args.local_rank,
    find_unused_parameters=True,
)

# 损失函数
criterion = nn.CrossEntropyLoss().cuda()

# 优化器
optimizer = torch.optim.SGD(
    model.parameters(), 
    args.lr, 
    momentum=args.momentum, 
    weight_decay=args.weight_decay
)

# 数据集
train_dataset = Cityscaples()

# 分配数据
train_sampler = DistributedSampler(train_dataset)

# 数据加载器
train_loader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=args.workers,
    pin_memory=True,
    sampler=train_sampler,
)


# 启动训练程序
# bash dst.sh



# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
