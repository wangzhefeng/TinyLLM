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
# ***************************************************

__all__ = []

# python libraries
import os
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)

from typing import Any
import urllib.request

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from utils.log_util import logger

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


def data_download(url: str, file_path: str):
    """
    data download
    data url: "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"
    """
    # version 1
    urllib.request.urlretrieve(url, file_path)
    # version 2
    '''
    # download
    with urllib.request.urlopen(url) as response:
        text_data = response.read().decode("utf-8")
    # write
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(text_data)
    '''


def data_load(url: str = None, data_path: str = "./dataset/pretrain/gpt", data_file: str = "the-verdict.txt"):
    """
    data load
    """
    # 数据文件路径
    if url is not None:
        file_path = Path(data_path).joinpath(url.split("/")[-1])
    else:
        file_path = Path(data_path).joinpath(data_file)
    # 数据下载、数据加载
    if not Path(file_path).exists():
        # download
        logger.info(f"Download data...")
        data_download(url, file_path)
        logger.info(f"Data has downloaded into '{data_path}'")
        # read
        logger.info(f"Load data...")
        with open(file_path, "r", encoding="utf-8") as file:
            raw_text = file.read()
        logger.info(f"Total number of character: {len(raw_text)}")
    else:
        # logger.info(f"Load data...")
        with open(file_path, "r", encoding="utf-8") as f:
            raw_text = f.read()
        # logger.info(f"Total number of character: {len(raw_text)}")

    return raw_text


def data_split(text: str, train_ratio: float=0.8):
    """
    数据读取
    训练、验证数据分割
    """
    split_idx = int(train_ratio * len(text))
    train_data = text[:split_idx]
    valid_data = text[split_idx:]
    
    return train_data, valid_data


class LLMDataset(Dataset):
    
    def __init__(self, text: str, tokenizer: Any, max_length: int, stride: int):
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
    
    def __getitem__(self, index: int):
        return self.input_ids[index], self.target_ids[index]


def create_dataloader(data_path: str, 
                      data_file: str, 
                      flag: str, 
                      train_ratio: float, 
                      tokenizer: Any,
                      batch_size: int=4, 
                      max_length: int=256, 
                      stride: int=128, 
                      num_workers: bool=0):
    # data load
    raw_text = data_load(data_path=data_path, data_file=data_file)
    # data load and split
    train_data, valid_data = data_split(text=raw_text, train_ratio=train_ratio)
    # data split
    assert flag in ['train', 'valid']
    text = train_data if flag == 'train' else valid_data
    # params
    shuffle = True if flag == "train" else False
    drop_last = False
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
        pin_memory=True,
        shuffle=shuffle,
        drop_last=drop_last,
        sampler=DistributedSampler(dataset),
        num_workers=num_workers,
    )
    
    return dataset, dataloader




# 测试代码 main 函数
def main():
    data_path = "./dataset/pretrain/gpt"
    data_file = "the-verdict.txt"

    # llm pretrain data
    raw_text = data_load(data_path = data_path, data_file = data_file)
    logger.info(type(raw_text))
    logger.info(f"len(raw_text): {len(raw_text)}")
    logger.info(f"raw_text[:99]: {raw_text[:99]}")
    logger.info(f"raw_text[:99]: {raw_text[-99:]}")

    # tokenizer
    from layers.tokenizers.tokenization import choose_tokenizer
    tokenizer = choose_tokenizer(tokenizer_model = "gpt2")         

    # data 
    train_data, valid_data = data_split(text = raw_text, train_ratio = 0.8)
    logger.info(f"train_data: {train_data[:100]} \ntrain_data length: {len(train_data)}")
    logger.info(f"valid_data: {valid_data[:10]} \nvalid_data length: {len(valid_data)}")
    
    # dataloader
    train_dataset, train_dataloader = create_dataloader(
        data_path = data_path, 
        data_file = data_file,
        flag = "train", 
        train_ratio = 0.8,
        tokenizer = tokenizer,
        batch_size = 4,
        max_length = 256,
        stride = 128,
        num_workers = 0,
    )
    for input, label in train_dataloader:
        logger.info(f"input: \n{input} \ninput.shape: {input.shape}")
        logger.info(f"label: \n{label} \nlabel.shape: {label.shape}")

if __name__ == "__main__":
    main()
