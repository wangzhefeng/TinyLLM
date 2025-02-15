# -*- coding: utf-8 -*-

# ***************************************************
# * File        : data_loader_finetuning.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2025-02-15
# * Version     : 0.1.021521
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import os
import sys
ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import tiktoken

from utils.log_util import logger

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


class SpamDataset(Dataset):
    
    def __init__(self, csv_file, tokenizer, max_length=None, pad_token_id=50256):
        # load data
        self.data = pd.read_csv(csv_file)
        # pre-tokenize texts
        self.encoded_texts = [
            tokenizer.encode(text)
            for text in self.data["Text"]
        ]
        # max length
        if max_length is None:
            self.max_length = self._longest_encoded_length()
        else:
            self.max_length = max_length
            # truncate sequences if they are longer than max_length
            self.encoded_texts = [
                encoded_text[:self.max_length]
                for encoded_text in self.encoded_texts
            ]
        # pad sequences to the longest sequence
        self.encoded_texts = [
            encoded_text + [pad_token_id] * (self.max_length - len(encoded_text))
            for encoded_text in self.encoded_texts
        ]
    
    def __getitem__(self, index):
        encoded = self.encoded_texts[index]
        label = self.data.iloc[index]["Label"]
        return (
            torch.tensor(encoded, dtype=torch.long),
            torch.tensor(label, dtype=torch.long)
        )
    
    def __len__(self):
        return len(self.data)

    def _longest_encoded_length(self):
        max_length = 0
        for encoded_text in self.encoded_texts:
            encoded_length = len(encoded_text)
            if encoded_length > max_length:
                max_length = encoded_length
        
        return max_length
        # Note: A more pythonic version to implement this method
        # is the following, which is also used in the next chapter:
        # return max(len(encoded_text) for encoded_text in self.encoded_texts)


def create_dataloader(data_path, 
                      max_length=None, 
                      batch_size=8, 
                      shuffle=False, 
                      drop_last=True, 
                      num_workers=0):
    # tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")
    # data set
    dataset = SpamDataset(
        csv_file = data_path,
        tokenizer=tokenizer,
        max_length=max_length,
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
    tokenizer = tiktoken.get_encoding("gpt2")
    logger.info(f"special token '<|endoftext|>' id: {tokenizer.encode('<|endoftext|>', allowed_special={'<|endoftext|>'})}")

    extracted_path = os.path.join(ROOT, r"dataset\finetuning\sms_spam_collection")
    batch_size = 8
    num_workers = 0

    train_dataset, train_loader = create_dataloader(
        data_path = os.path.join(extracted_path, "train.csv"),
        max_length = None,
        batch_size = batch_size,
        shuffle = True,
        drop_last = True,
    )
    valid_dataset, valid_loader = create_dataloader(
        data_path = os.path.join(extracted_path, "valid.csv"),
        max_length = train_dataset.max_length,
        batch_size = batch_size,
        shuffle = False,
        drop_last = False,
    )
    test_dataset, test_loader = create_dataloader(
        data_path = os.path.join(extracted_path, "test.csv"),
        max_length = train_dataset.max_length,
        batch_size = batch_size,
        shuffle = False,
        drop_last = False,
    )
    logger.info(f"train_dataset.max_length: {train_dataset.max_length}")
    logger.info(f"valid_dataset.max_length: {valid_dataset.max_length}")
    logger.info(f"test_dataset.max_length: {test_dataset.max_length}")

    logger.info(f"Train loader:")
    for input_batch, target_batch in train_loader:
        pass
    logger.info(f"Input batch dim: {input_batch.shape}")
    logger.info(f"Target batch dim: {target_batch.shape}")

    logger.info(f"{len(train_loader)} training batches")
    logger.info(f"{len(valid_loader)} validation batches")
    logger.info(f"{len(test_loader)} test batches")

if __name__ == "__main__":
    main()
