# -*- coding: utf-8 -*-

# ***************************************************
# * File        : data_loader_finetuning.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-02-15
# * Version     : 0.1.021521
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import os
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)


import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from utils.log_util import logger

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


class SpamDataset(Dataset):
    
    def __init__(self, data_path, tokenizer, pad_token_id, max_length=None):
        # load csv data
        self.data = pd.read_csv(data_path)
        logger.info(f"{data_path.split('/')[-1]}: \n{self.data.head()}")
        
        # pre-tokenize texts
        self.encoded_texts = [tokenizer.encode(text) for text in self.data["Text"]]
        # logger.info(f"raw encoded_texts[0]: {self.encoded_texts[0]}")
        
        # max length of encoded texts
        if max_length is None:
            # pad all messages to the length of the longest message in the dataset or batch
            self.max_length = self._longest_encoded_length()
        else:
            # truncate sequences if they are longer than max_length
            self.max_length = max_length
            self.encoded_texts = [encoded_text[:self.max_length] for encoded_text in self.encoded_texts]
        
        # pad sequences to the longest sequence with the pad token
        self.encoded_texts = [
            encoded_text + [pad_token_id] * (self.max_length - len(encoded_text))
            for encoded_text in self.encoded_texts
        ]
        # logger.info(f"padded encoded_texts[0]: {self.encoded_texts[0]}")
    
    def __getitem__(self, index):
        encoded = self.encoded_texts[index]
        label = self.data.iloc[index]["Label"]
        return (
            torch.tensor(encoded, dtype = torch.long),
            torch.tensor(label, dtype = torch.long)
        )
    
    def __len__(self):
        return len(self.data)

    def _longest_encoded_length(self):
        return max(len(encoded_text) for encoded_text in self.encoded_texts)


def create_dataloader(data_path, 
                      max_length = None, 
                      batch_size = 8, 
                      shuffle = False, 
                      drop_last = True, 
                      tokenizer = None,
                      pad_token_id = None,
                      num_workers = 0):
    # data set
    dataset = SpamDataset(
        data_path = data_path,
        tokenizer = tokenizer,
        pad_token_id = pad_token_id,
        max_length = max_length,
    )
    # data loader
    dataloader = DataLoader(
        dataset = dataset,
        batch_size = batch_size,
        shuffle = shuffle,
        drop_last = drop_last,
        num_workers = num_workers,
    )

    return dataset, dataloader




# 测试代码 main 函数
def main():
    from data_provider.finetune.text_clf.data_config import data_dir
    import tiktoken

    # params
    batch_size = 8
    # tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")
    # pad token
    pad_token_id = tokenizer.encode("<|endoftext|>", allowed_special = {"<|endoftext|>"})[0]

    # dataset and dataloader
    train_dataset, train_loader = create_dataloader(
        data_path = Path(data_dir).joinpath("train.csv"),
        max_length = None,
        batch_size = batch_size,
        shuffle = True,
        drop_last = True,
        tokenizer = tokenizer,
        pad_token_id = pad_token_id,
    )
    valid_dataset, valid_loader = create_dataloader(
        data_path = Path(data_dir).joinpath("valid.csv"),
        max_length = train_dataset.max_length,
        batch_size = batch_size,
        shuffle = False,
        drop_last = False,
        tokenizer = tokenizer,
        pad_token_id = pad_token_id,
    )
    test_dataset, test_loader = create_dataloader(
        data_path = Path(data_dir).joinpath("test.csv"),
        max_length = train_dataset.max_length,
        batch_size = batch_size,
        shuffle = False,
        drop_last = False,
        tokenizer = tokenizer,
        pad_token_id = pad_token_id,
    )
    logger.info(f"train_dataset.max_length: {train_dataset.max_length}")
    logger.info(f"valid_dataset.max_length: {valid_dataset.max_length}")
    logger.info(f"test_dataset.max_length: {test_dataset.max_length}")
    logger.info(f"training batches: {len(train_loader)}")
    logger.info(f"validation batches: {len(valid_loader)}")
    logger.info(f"test batches: {len(test_loader)}")

    # dataloader test
    logger.info(f"Train loader:")
    for input_batch, target_batch in train_loader:
        pass
    logger.info(f"input_batch: \n{input_batch} \nInput batch dim: {input_batch.shape}")
    logger.info(f"target_batch: \n{target_batch} \nTarget batch dim: {target_batch.shape}") 

if __name__ == "__main__":
    main()
