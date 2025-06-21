# -*- coding: utf-8 -*-

# ***************************************************
# * File        : data_factory.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-05-03
# * Version     : 1.0.050303
# * Description : description
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


from torch.utils.data import DataLoader, DistributedSampler

from minimind.data_provider.data_loader import PretrainDataset

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


def data_provider(args, tokenizer, ddp):
    # dataset
    train_data = PretrainDataset(
        args.data_path, 
        tokenizer, 
        max_length = args.max_seq_len
    )
    # sampler
    train_sampler = DistributedSampler(train_data) if ddp else None
    # data loader
    train_loader = DataLoader(
        train_data,
        batch_size = args.batch_size,
        pin_memory = True,
        drop_last = False,
        shuffle = False,
        num_workers = args.num_workers,
        sampler = train_sampler,
    )

    return train_data, train_loader




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
