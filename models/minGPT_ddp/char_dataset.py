import fsspec
import torch
from torch.utils.data import Dataset
from dataclasses import dataclass

from utils.log_util import logger

"""
Adapted from https://github.com/karpathy/minGPT/blob/master/projects/chargpt/chargpt.py
"""

@dataclass
class DataConfig:
    path: str = None
    block_size: int = None
    train_split: float = None
    truncate: float = 1.0


class CharDataset(Dataset):

    def __init__(self, config: DataConfig): 
        self.config = config

        # load data
        data = self._load_data()
        data_size = len(data)
        # split data into chars
        chars = sorted(list(set(data)))
        vocab_size = len(chars)
        logger.info(f'Data has {data_size} characters, {vocab_size} unique.')
        # tokenize
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        # data config
        self.block_size = config.block_size
        self.vocab_size = vocab_size
        self.data = data

    def _load_data(self):
        data = fsspec.open(self.config.path).open().read().decode('utf-8')
        data = data[:int(len(data) * self.config.truncate)]

        return data

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        # grab a chunk of (block_size + 1) characters from the data
        chunk = self.data[idx:idx + self.block_size + 1]
        # encode every character to an integer
        dix = [self.stoi[s] for s in chunk]
        # return as tensors
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)

        return x, y




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
