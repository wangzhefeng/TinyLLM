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

import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


class LLMDataset(Dataset):
    
    def __init__(self, text: str, tokenizer, max_length: int, stride: int):
        self.input_ids = []
        self.target_ids = []

        # tokenize the entrie text
        token_ids = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
        
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


def create_dataloader(text, 
                      batch_size=4, 
                      max_length=256, 
                      stride=128, 
                      shuffle=True, 
                      drop_last=True, 
                      num_workers=0):
    # initialize the tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")
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
