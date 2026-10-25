# -*- coding: utf-8 -*-

# ***************************************************
# * File        : data_loader.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-02-16
# * Version     : 0.1.021619
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)
from functools import partial

import torch
from torch.utils.data import Dataset, DataLoader

from utils.load_save_data import load_json_data
from layers.tokenizers.chat import format_input_alpaca
from utils.log_util import logger

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


class InstructionDataset(Dataset):
    
    def __init__(self, data, tokenizer):
        # load data
        self.data = data
        # pre-tokenize texts
        self.encoded_texts = []
        for entry in self.data:
            # format input into instruction-response template
            # （与书中 ch07 模板一致：### Response:\n，冒号后无空格）
            instruction_plus_input = format_input_alpaca(entry)
            response_text = f"\n\n### Response:\n{entry['output']}"
            full_text = instruction_plus_input + response_text
            # convert instruction-response entry into token IDs
            token_ids = tokenizer.encode(full_text)
            # collect token IDs
            self.encoded_texts.append(token_ids)
    
    def __getitem__(self, index):
        return self.encoded_texts[index]
    
    def __len__(self):
        return len(self.data)


class InstructionDatasetWithMasking(Dataset):
    """
    指令数据集（prompt-masking 变体，LLMs-from-scratch ch07 exercise_experiments.py
    的 InstructionDatasetWithMasking）。

    额外记录每条样本的 instruction（含 input）token 长度，
    __getitem__ 返回 (instruction_length, token_ids) 二元组，
    供 custom_collate_fn_mask_out_instruction 在 targets 中掩掉 prompt 部分。
    """

    def __init__(self, data, tokenizer):
        # load data
        self.data = data
        # instruction token 长度（与 encoded_texts 按下标对齐）
        self.instruction_lengths = []
        # pre-tokenize texts
        self.encoded_texts = []
        for entry in self.data:
            instruction_plus_input = format_input_alpaca(entry)
            response_text = f"\n\n### Response:\n{entry['output']}"
            full_text = instruction_plus_input + response_text
            # convert instruction-response entry into token IDs
            token_ids = tokenizer.encode(full_text)
            # collect token IDs and instruction length
            self.encoded_texts.append(token_ids)
            self.instruction_lengths.append(len(tokenizer.encode(instruction_plus_input)))

    def __getitem__(self, index):
        # 返回 (instruction_length, token_ids)，供 masking collate 使用
        return self.instruction_lengths[index], self.encoded_texts[index]

    def __len__(self):
        return len(self.data)


def custom_collate_fn(batch, 
                      pad_token_id = 50256, 
                      ignore_index = -100, 
                      allowed_max_length = None, 
                      device = "cpu"):
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


def custom_collate_fn_mask_out_instruction(batch, 
                                           pad_token_id = 50256, 
                                           ignore_index = -100, 
                                           allowed_max_length = None, 
                                           device = "cpu"):
    """
    prompt-masking collate（LLMs-from-scratch ch07 exercise_experiments.py 的
    custom_collate_with_masking_fn 忠实移植）。

    与 custom_collate_fn 的区别：batch 中的每个 item 是
    (instruction_length, token_ids) 二元组（来自 InstructionDatasetWithMasking），
    targets 中属于 instruction/input（prompt）的部分被置为 ignore_index=-100，
    只对 response（含首个 <|endoftext|>）计算损失。

    注意：instruction_length 是 prompt 单独编码的长度，BPE 跨界合并可能带来
    ±1 token 的边界近似（与书一致的教学近似）。
    """
    # Find the longest sequence in the batch（batch 元素为 (instruction_length, token_ids)）
    batch_max_length = max(len(item) + 1 for _, item in batch)

    # Pad and prepare inputs and targets
    inputs_lst, targets_lst = [], []

    for instruction_length, item in batch:
        new_item = item.copy()
        # Add an <|endoftext|> token
        new_item += [pad_token_id]
        # Pad sequences to max_length
        padded = new_item + [pad_token_id] * (batch_max_length - len(new_item))
        inputs = torch.tensor(padded[:-1])  # Truncate the last token for inputs
        targets = torch.tensor(padded[1:])  # Shift +1 to the right for targets

        # Replace all but the first padding tokens in targets by ignore_index
        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index

        # Mask all input and instruction tokens in the targets
        # （targets 已整体右移一位，故 prompt 部分对应 targets[:instruction_length-1]）
        targets[:instruction_length - 1] = ignore_index

        # Optionally truncate to maximum sequence length
        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]

        inputs_lst.append(inputs)
        targets_lst.append(targets)

    # Convert list of inputs and targets to tensors and transfer to target device
    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)

    return inputs_tensor, targets_tensor


def load_split_data(data_path: str, train_ratio: int = 0.85, test_ratio: int = 0.10):
    """
    data split
    """
    # data load
    data = load_json_data(data_path = data_path)
    # logger.info(f"data type: {type(data)}")
    # data split ratio
    train_portion = int(len(data) * train_ratio)
    test_portion = int(len(data) * test_ratio)
    valid_portion = len(data) - train_portion - test_portion
    # data split
    train_data = data[:train_portion]
    test_data = data[train_portion:(train_portion + test_portion)]
    valid_data = data[(train_portion + test_portion):]
    
    return train_data, test_data, valid_data


def create_dataloader(data, 
                      device, 
                      tokenizer, 
                      pad_token_id = 50256,
                      ignore_index = -100,
                      allowed_max_length = 1024,
                      batch_size = 8, 
                      shuffle = True, 
                      drop_last = True, 
                      num_workers = 0,
                      mask_out_instruction = False):
    """
    Args:
        mask_out_instruction: True 时使用 InstructionDatasetWithMasking +
            custom_collate_fn_mask_out_instruction（prompt 部分不计损失，
            对应书 ch07 exercise 的 instruction-masking 实验）
    """
    # collate function 与 dataset 按 mask 开关二选一
    if mask_out_instruction:
        collate_fn = partial(
            custom_collate_fn_mask_out_instruction, 
            pad_token_id = pad_token_id, 
            ignore_index = ignore_index,
            allowed_max_length = allowed_max_length, 
            device = device
        )
        dataset = InstructionDatasetWithMasking(data = data, tokenizer = tokenizer)
    else:
        collate_fn = partial(
            custom_collate_fn, 
            pad_token_id = pad_token_id, 
            ignore_index = ignore_index,
            allowed_max_length = allowed_max_length, 
            device = device
        )
        # data set
        dataset = InstructionDataset(data = data, tokenizer = tokenizer)
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
    """
    # ------------------------------
    # collate_fn test
    # ------------------------------
    # batch
    inputs_1 = [0, 1, 2, 3, 4]
    inputs_2 = [5, 6]
    inputs_3 = [7, 8, 9]
    batch = (inputs_1, inputs_2, inputs_3)
    logger.info(f"batch: {batch}")
    
    # batch padding
    collate_fn = partial(collate_fn, device = device, allowed_max_length = 1024)
    inputs, targets = collate_fn(batch)
    logger.info(f"inputs: \n{inputs}")
    logger.info(f"targets: \n{targets}")
    """
    # ------------------------------
    # data loader test
    # ------------------------------
    # data load and split
    from post_training.sft.data_provider.instruction_follow import data_config 
    train_data, test_data, valid_data = load_split_data(data_config.data_path)
    # device
    from utils.device import device_setting
    # device
    device = device_setting()

    # tokenizer
    import tiktoken
    tokenizer = tiktoken.get_encoding("gpt2") 

    # dataset and dataloader
    torch.manual_seed(123)
    train_dataset, train_dataloader = create_dataloader(
        data = train_data,
        device = device,
        tokenizer = tokenizer,
        pad_token_id = 50256,
        ignore_index = -100,
        allowed_max_length = 1024,
        batch_size = 8,
        shuffle = True,
        drop_last = True,
        num_workers = 0,
    )
    test_dataset, test_dataloader = create_dataloader(
        data = test_data,
        device = device,
        tokenizer = tokenizer,
        pad_token_id = 50256,
        ignore_index = -100,
        allowed_max_length = 1024,
        batch_size = 8,
        shuffle = False,
        drop_last = False,
        num_workers = 0,
    )
    valid_dataset, valid_dataloader = create_dataloader(
        data = valid_data,
        device = device,
        tokenizer = tokenizer,
        pad_token_id = 50256,
        ignore_index = -100,
        allowed_max_length = 1024,
        batch_size = 8,
        shuffle = False,
        drop_last = False,
        num_workers = 0,
    )
    logger.info(f"Training data length: {len(train_data)} Training batches: {len(train_dataloader)}")
    logger.info(f"Test data length: {len(test_data)} Test batches: {len(test_dataloader)}")
    logger.info(f"Valid data length: {len(valid_data)} Validation batches: {len(valid_dataloader)}")

    # test
    logger.info(f"Train loader:")
    for batch, (inputs, targets) in enumerate(train_dataloader):
        # logger.info(f"batch: {batch} inputs: \n{inputs[7]}")
        # logger.info(f"batch: {batch} targets: \n{targets[7]}")
        logger.info(f"batch: {batch} inputs.shape: {inputs.shape}, targets.shape: {targets.shape}")

if __name__ == "__main__":
    main()
