# -*- coding: utf-8 -*-

# ***************************************************
# * File        : data_sampling.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-01-23
# * Version     : 1.0.012323
# * Description : https://github.com/rasbt/LLMs-from-scratch/blob/2dc46bedc6e86b79a16c4099e557564cd23e03ef/ch02/04_bonus_dataloader-intuition/dataloader-intuition.ipynb
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
from pathlib import Path

import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader

from data_provider.pretrain.data_load import data_load

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


def data_loader(data_path: str, data_file: str, train_ratio: float=0.8):
    """
    训练、验证数据分割
    """
    # load data
    text = data_load(data_path, data_file)
    # split data
    split_idx = int(train_ratio * len(text))
    train_data = text[:split_idx]
    valid_data = text[split_idx:]
    
    return train_data, valid_data


class LLMDataset(Dataset):
    
    def __init__(self, text: str, tokenizer, max_length: int, stride: int):
        self.input_ids = []
        self.target_ids = []

        # tokenize the entrie text
        token_ids = tokenizer.encode(text)
        # use a sliding window to chunk the book into overlapping sequences of max_length
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, index):
        return self.input_ids[index], self.target_ids[index]


def create_dataloader(data_path, 
                      data_file, 
                      train_ratio, 
                      flag, 
                      tokenizer,
                      batch_size=4, 
                      max_length=256, 
                      stride=128, 
                      shuffle=True, 
                      drop_last=True, 
                      num_workers=0):
    # data split
    train_data, valid_data = data_loader(data_path, data_file, train_ratio)
    assert flag in ['train', 'valid']
    text = train_data if flag == 'train' else valid_data
    # create dataset
    dataset = LLMDataset(
        text=text,
        tokenizer=tokenizer, 
        max_length=max_length, 
        stride=stride
    )
    # create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )
    
    return dataloader




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
