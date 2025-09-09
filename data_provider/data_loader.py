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
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)
from typing import Any

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from data_provider.load_save_data import (
    load_local_data, 
    load_hf_data,
)
from utils.ddp_utils import is_dist
from utils.log_util import logger

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


def data_split(text: str, train_ratio: float=0.8):
    """
    训练、验证数据分割
    """
    split_idx = int(train_ratio * len(text))
    train_data = text[:split_idx]
    valid_data = text[split_idx:]
    
    return train_data, valid_data


class LLMDataset(Dataset):
    
    def __init__(self, text: str, tokenizer: Any, max_len: int, stride: int, num_examples: int, flag: str, dtype, device):
        # sampling stride
        stride = stride if stride is not None else max_len + 1
        
        # tokenize the entrie text
        logger.info("Tokenization...")
        token_ids = tokenizer.encode(text)
        assert len(token_ids) > max_len, "Number of tokenized inputs must at least be equal to max_len+1"
        logger.info(f"{flag.capitalize()} token length: {len(token_ids)}")

        # Data sampling with a sliding window to chunk the text into overlapping sequences of max_len
        self.input_blocks = []
        self.target_blocks = []
        for i in range(0, len(token_ids), stride):
            # sampling block
            block = token_ids[i:(i + max_len + 1)]

            # Skip blocks that are too short
            if len(block) < max_len + 1:
                continue

            # sampling input and target
            input_seq =  block[:-1]
            target_seq = block[1:]
            # logger.info(f"debug::i: {i}")
            # logger.info(f"debug::input_seq  start_idx:end_idx: {i}:{(i + max_len)}")
            # logger.info(f"debug::target_seq start_idx:end_idx: {i+1}:{(i + max_len + 1)}")

            # input and target block
            self.input_blocks.append(input_seq)
            self.target_blocks.append(target_seq)
            # self.input_blocks.append(torch.tensor(input_seq))
            # self.target_blocks.append(torch.tensor(target_seq))

            # Stop if have enough examles
            if len(self.input_blocks) > num_examples:
                break
        # Convert to tensors for PyTorch and move to device
        self.input_blocks = torch.tensor(self.input_blocks, dtype=dtype)
        self.target_blocks = torch.tensor(self.target_blocks, dtype=dtype)
        if device:
            self.input_blocks = self.input_blocks.to(device)
            self.target_blocks = self.target_blocks.to(device)
    
    def __len__(self):
        return len(self.input_blocks)
    
    def __getitem__(self, index: int):
        input_blocks = self.input_blocks[index]
        target_blocks = self.target_blocks[index]

        return input_blocks, target_blocks


def create_dataloader(data_source: str,  # option: ["huggingface", "local"]
                      url: str,
                      data_path: str,
                      data_file: str,
                      flag: str,
                      train_ratio: float,
                      tokenizer: Any,
                      batch_size: int=4,
                      max_len: int=256,
                      stride: int=None,
                      num_workers: bool=0,
                      num_examples: int=100000,
                      dtype=torch.long,
                      device=None):
    # data load
    assert data_source in ['huggingface', 'local'], "data_source must be in ['huggingface', 'local']"
    if data_source == "local":
        raw_text = load_local_data(url, data_path=data_path, data_file=data_file)    
    elif data_source == "huggingface":
        raw_text = load_hf_data(data_path=data_path, data_name=data_file, cache_dir="./dataset/pretrain")
    logger.info(f"Train data character length: {len(raw_text)}")
    # data split
    train_data, valid_data = data_split(text=raw_text, train_ratio=train_ratio)
    assert flag in ['train', 'valid']
    text = train_data if flag == 'train' else valid_data
    # params
    shuffle = True if flag == "train" else False
    drop_last = False
    # create dataset
    dataset = LLMDataset(
        text=text, 
        tokenizer=tokenizer, 
        max_len=max_len, 
        stride=stride,
        num_examples=num_examples,
        flag=flag,
        dtype=dtype,
        device=device,
    )
    # create dataloader
    if is_dist():
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            pin_memory=True,
            shuffle=False,
            drop_last=drop_last,
            sampler=DistributedSampler(dataset),
            num_workers=num_workers,
        )
    else:
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            pin_memory=True,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers,
        )
    
    return dataset, dataloader




# 测试代码 main 函数
def main():
    # tokenizer
    from layers.tokenizers.tokenization import choose_tokenizer
    tokenizer = choose_tokenizer(tokenizer_model = "tiktoken_gpt2_bpe")
    # ------------------------------
    # the-verdict.txt
    # ------------------------------
    # data path
    data_source = "local"
    url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"
    data_path = "./dataset/pretrain/gpt"
    data_file = "the-verdict.txt" 
    # dataloader
    train_dataset, train_dataloader = create_dataloader(
        data_source=data_source,
        url = url,
        data_path = data_path, 
        data_file = data_file,
        flag = "train", 
        train_ratio = 0.8,
        tokenizer = tokenizer,
        batch_size = 4,
        max_len = 256,
        num_workers = 0,
    )
    for input, label in train_dataloader:
        logger.info(f"input: \n{input} \ninput.shape: {input.shape}")
        logger.info(f"label: \n{label} \nlabel.shape: {label.shape}")
        break
    # ------------------------------
    # pg145.txt
    # ------------------------------
    # data path
    data_source = "local"
    url = "https://www.gutenberg.org/cache/epub/145/pg145.txt"
    data_path = "./dataset/pretrain/gpt"
    data_file = "pg145.txt" 
    # dataloader
    train_dataset, train_dataloader = create_dataloader(
        data_source=data_source,
        url = url,
        data_path = data_path, 
        data_file = data_file,
        flag = "train", 
        train_ratio = 0.8,
        tokenizer = tokenizer,
        batch_size = 4,
        max_len = 256,
        num_workers = 0,
    )
    for input, label in train_dataloader:
        logger.info(f"input: \n{input} \ninput.shape: {input.shape}")
        logger.info(f"label: \n{label} \nlabel.shape: {label.shape}")
        break
    # ------------------------------
    # huggingface data
    # ------------------------------
    # data path
    data_source="huggingface"
    url = None
    data_path="EleutherAI/wikitext_document_level"
    data_file="wikitext-2-raw-v1"
    # dataloader
    valid_dataset, valid_dataloader = create_dataloader(
        data_source = "huggingface",
        url=None,
        data_path = data_path, 
        data_file = data_file,
        flag = "valid", 
        train_ratio = 0.8,
        tokenizer = tokenizer,
        batch_size = 4,
        max_len = 256,
        num_workers = 0,
    )
    for input, label in valid_dataloader:
        logger.info(f"input: \n{input} \ninput.shape: {input.shape}")
        logger.info(f"label: \n{label} \nlabel.shape: {label.shape}")
        break

if __name__ == "__main__":
    main()
