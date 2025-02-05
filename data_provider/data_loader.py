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
ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from utils.log_util import logger

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
    import tiktoken
    tokenizer = tiktoken.get_encoding("gpt2")
    # create dataset
    dataset = LLMDataset(text, tokenizer, max_length, stride)
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
    import tiktoken
    from data_provider.data_load_pretrain import data_load

    # ------------------------------
    # data download & load
    # ------------------------------
    raw_text = data_load(
        url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"
    )
    
    # ------------------------------
    # data sampling
    # ------------------------------
    # tokenization
    tokenizer = tiktoken.get_encoding("gpt2")
    enc_text = tokenizer.encode(raw_text) 
    logger.info(f"Pre-train token size: {len(enc_text)}")
    
    # token id
    enc_sample = enc_text[50:]

    # context size
    context_size = 4
 
    # data sampling
    x = enc_sample[:context_size]
    y = enc_sample[1:context_size]
    logger.info(f"x: {x}")
    logger.info(f"y:      {y}")
    
    # sliding window token ids
    for i in range(1, context_size+1):
        context = enc_sample[:i]
        desired = enc_sample[i]
        logger.info(f"{context} ----> {desired}")

    # sliding window tokens
    for i in range(1, context_size+1):
        context = enc_sample[:i]
        desired = enc_sample[i]
        logger.info(f"{tokenizer.decode(context)} ----> {tokenizer.decode([desired])}")
    
    # ------------------------------
    # dataset and dataloader test
    # ------------------------------
    # params
    batch_size = 8
    max_length = 4
    # data loader
    dataloader = create_dataloader(
        raw_text,
        batch_size=batch_size,
        max_length=max_length,
        stride=max_length,
        shuffle=False,
        drop_last=True,
    )
    # data loader test
    for batch in dataloader:
        # data batch
        x, y = batch 
        logger.info(f"x: \n{x}")
        logger.info(f"y: \n{y}")
        break

if __name__ == "__main__":
    main()
