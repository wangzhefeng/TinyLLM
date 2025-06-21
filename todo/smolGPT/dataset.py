# -*- coding: utf-8 -*-

# ***************************************************
# * File        : dataset.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-01-31
# * Version     : 1.0.013122
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# * TODO        : 1.
# ***************************************************

__all__ = []

# python libraries
import os
import sys
ROOT = str(os.getcwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)
import glob
import random
from typing import Iterator, Tuple
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import IterableDataset, DataLoader

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


class PreTokDataset(IterableDataset):
    
    def __init__(self, split: str, max_seq_len: int):
        super().__init__()
        self.split = split
        self.max_seq_len = max_seq_len
    
    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        bin_dir = Path("data/TinyStories_all_data")
        shard_filenames = sorted(glob.glob(str(bin_dir / "*.bin")))
        shard_filenames = (shard_filenames[1:] if self.split == "train" else shard_filenames[:1])

        rng = random.Random(42)
        while True:
            rng.shuffle(shard_filenames)
            for shard in shard_filenames:
                data = np.memmap(shard, dtype=np.uint16, mode="r")
                num_batches = len(data) // self.max_seq_len - 1
                idxs = list(range(num_batches))
                rng.shuffle(idxs)

                for idx in idxs:
                    start = idx * self.max_seq_len
                    end = (idx + 1) * self.max_seq_len
                    chunk = torch.from_numpy(data[start:end].astype(np.int64))
                    x = chunk[:-1]
                    y = chunk[1:]
                    yield x, y


class Task:

    @staticmethod
    def iter_batches(
        batch_size: int, device: str, num_workers: int = 0, **dataset_kwargs
    ) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        ds = PreTokDataset(**dataset_kwargs)
        dl = DataLoader(ds, batch_size=batch_size, num_workers=num_workers)
        for x, y in dl:
            x = x.to(device)
            y = y.to(device)
            yield x, y




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
