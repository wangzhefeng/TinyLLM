# -*- coding: utf-8 -*-

# ***************************************************
# * File        : data_loader.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2025-02-16
# * Version     : 0.1.021619
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
from functools import partial

import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader

from finetuning.instruction_follow.data_load import load_file
from finetuning.instruction_format import format_input_alpaca
from utils.device import device
from utils.log_util import logger

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


class InstructionDataset(Dataset):
    
    def __init__(self, data, tokenizer):
        self.data = data
        # pre-tokenize texts
        self.encoded_texts = []
        for entry in data:
            # format input into instruction-response template
            full_text = format_input_alpaca(entry)
            # convert instruction-response entry into token IDs
            token_ids = tokenizer.encode(full_text)
            
            self.encoded_texts.append(token_ids)
    
    def __getitem__(self, index):
        return self.encoded_texts[index]

    def __len__(self):
        return len(self.data)


def collate_fn(batch, pad_token_id=50256, ignore_index=-100, allowed_max_length=None, device="cuda"):
    # Find the longest sequence in the batch
    batch_max_length = max(len(item) + 1 for item in batch)
    
    # Pad and prepare inputs and targets
    inputs_lst, targets_lst = [], []
    for item in batch:
        # item
        new_item = item.copy() 
        # Add an <|endoftext|> token
        new_item += [pad_token_id]
        # Pad sequences to max_length
        padded = new_item + [pad_token_id] * (batch_max_length - len(new_item))
        # inputs and targets
        inputs = torch.tensor(padded[:-1])  # Truncate the last token for inputs
        targets = torch.tensor(padded[1:])  # Shift +1 to the right for targets
        # Replace all but the first padding tokens in targets by ignore_index
        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index
        # Optionally truncate to maximum sequence length
        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]
        # collect inputs_lst, target_lst
        inputs_lst.append(inputs)
        targets_lst.append(targets)
    # Convert list of inputs and targets to tensors and transfer to target device
    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)

    return inputs_tensor, targets_tensor

collate_fn = partial(collate_fn, device = device, allowed_max_length = 1024)


def create_dataloader(data, batch_size = 8, shuffle = True, drop_last = True, num_workers = 0):
    # tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")
    # data set
    dataset = InstructionDataset(data = data, tokenizer=tokenizer)
    # data loader
    dataloader = DataLoader(
        dataset = dataset,
        batch_size = batch_size,
        collate_fn = collate_fn,
        shuffle = shuffle,
        drop_last = drop_last, 
        num_workers = num_workers,
    )

    return dataset, dataloader




# 测试代码 main 函数
def main():
    # batch
    inputs_1 = [0, 1, 2, 3, 4]
    inputs_2 = [5, 6]
    inputs_3 = [7, 8, 9]
    batch = (inputs_1, inputs_2, inputs_3)
    logger.info(f"batch: {batch}")
    
    # batch padding
    inputs, targets = collate_fn(batch)
    logger.info(f"inputs: \n{inputs}")
    logger.info(f"targets: \n{targets}")

    # data
    data = load_file(file_path = "./dataset/finetuning/instruction-data.json")

    # data split
    train_portion = int(len(data) * 0.85)
    test_portion = int(len(data) * 0.10)
    valid_portion = len(data) - train_portion - test_portion
    train_data = data[:train_portion]
    test_data = data[train_portion:(train_portion+test_portion)]
    valid_data = data[(train_portion+test_portion):]
    logger.info(f"train_data: \n{train_data[0]}")
    logger.info(f"test_data: \n{test_data[0]}")
    logger.info(f"valid_data: \n{valid_data[0]}")
    logger.info(f"Training data length: {len(train_data)}")
    logger.info(f"Test data length: {len(test_data)}")
    logger.info(f"Validation data length: {len(valid_data)}")

    # dataset and dataloader
    torch.manual_seed(123)
    train_dataset, train_dataloader = create_dataloader(
        data = train_data,
        batch_size = 8,
        shuffle = True,
        drop_last = True,
        num_workers = 0,
    )
    test_dataset, test_dataloader = create_dataloader(
        data = test_data,
        batch_size = 8,
        shuffle = False,
        drop_last = False,
        num_workers = 0,
    )
    valid_dataset, valid_dataloader = create_dataloader(
        data = valid_data,
        batch_size = 8,
        shuffle = False,
        drop_last = False,
        num_workers = 0,
    )

    # test
    logger.info(f"Train loader:")
    for inputs, targets in train_dataloader:
        # logger.info(f"inputs: \n{inputs[7]}")
        # logger.info(f"targets: \n{targets[7]}")
        logger.info(f"inputs.shap: {inputs.shape}, targets.shape: {targets.shape}")

if __name__ == "__main__":
    main()
