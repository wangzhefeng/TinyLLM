# -*- coding: utf-8 -*-

# ***************************************************
# * File        : data_loader.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2025-02-22
# * Version     : 0.1.022202
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

import torch
from torch.utils.data import Dataset, DataLoader

from data_provider.load_save_data import load_json_data
from data_provider.finetune import instruction_format
from utils.log_util import logger

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


class PreferenceDataset(Dataset):

    def __init__(self, data, tokenizer):
        # load data
        self.data = data
        # pre-tokenize texts
        self.encoded_texts = []
        for entry in data:
            # format input into instruction-response template, prompt tokenization
            prompt = instruction_format.format_input_alpaca(entry) 
            prompt_token_ids = tokenizer.encode(prompt)
            # logger.info(f"prompt: \n{prompt}")
            # logger.info(f"prompt_token_ids length: {len(prompt_token_ids)}")

            # responses
            rejected_response = entry["rejected"]
            chosen_response = entry["chosen"]

            # response format and tokenization
            chosen_full_text = f"{prompt}\n\n### Response:\n{chosen_response}"
            rejected_full_text = f"{prompt}\n\n### Response:\n{rejected_response}"
            chosen_full_token_ids = tokenizer.encode(chosen_full_text)
            rejected_full_token_ids = tokenizer.encode(rejected_full_text)
            # logger.info(f"chosen_full_text: \n{chosen_full_text}")
            # logger.info(f"rejected_full_text: \n{rejected_full_text}")
            # logger.info(f"chosen_full_token_ids length: {len(chosen_full_token_ids)}")
            # logger.info(f"rejected_full_token_ids length: {len(rejected_full_token_ids)}")

            # collect token IDs
            self.encoded_texts.append({
                "prompt": prompt_token_ids,
                "chosen": chosen_full_token_ids,
                "rejected": rejected_full_token_ids,
            })
        # logger.info(f"self.encoded_texts[0]: \n{self.encoded_texts[0]}")

    def __getitem__(self, index):
        return self.encoded_texts[index]

    def __len__(self):
        return len(self.data)


def custom_collate_fn(batch, 
                      pad_token_id = 50256,
                      allowed_max_length = None,
                      mask_prompt_tokens = True,
                      device="cpu"):
    # 初始化列表以保存批次数据
    batch_data = {
        "prompt": [],
        "chosen": [],
        "rejected": [],
        "chosen_mask": [],
        "rejected_mask": [],
    }
    # 确定最长的序列以设置相同的填充长度
    max_length_common = 0
    if batch:
        for key in ["chosen", "rejected"]:
            # logger.info(f"{50*'-'} key: {key}")
            current_max = max(len(item[key]) + 1 for item in batch)
            # logger.info(f"current_max: {current_max}")
            max_length_common = max(max_length_common, current_max)
            # logger.info(f"max_length_common: {max_length_common}")
    # logger.info(f"max_length_common: {max_length_common}")
    
    # 处理批次中的每个项目
    for item in batch:
        # logger.info(f"item: \n{item}")
        # 处理 prompt
        prompt = torch.tensor(item["prompt"])
        batch_data["prompt"].append(prompt)
        # 处理 chosen, rejected
        for key in ["chosen", "rejected"]:
            # 根据相同的最大长度调整填充
            sequence = item[key]
            padded = sequence + [pad_token_id] * (max_length_common - len(sequence))
            mask = torch.ones(len(padded)).bool()
            # 将所有填充标记的掩码设置为 False
            mask[len(sequence):] = False
            # 将所有填充标记的掩码设置为 False
            # +2 将 "### Response" 之前的 2 个换行符 ("\n") 标记设置为 False
            if mask_prompt_tokens:
                mask[:prompt.shape[0] + 2] = False
            # 保存填充好的 token ids
            batch_data[key].append(torch.tensor(padded))
            batch_data[f"{key}_mask"].append(mask)

    # 最终处理
    for key in ["chosen", "rejected", "chosen_mask", "rejected_mask"]:
        # 将所有序列堆叠为给定键的张量
        tensor_stack = torch.stack(batch_data[key])
        # 可选：截断到最大序列长度
        if allowed_max_length is not None:
            tensor_stack = tensor_stack[:, :allowed_max_length]
        # 移动到指定设备
        batch_data[key] = tensor_stack.to(device)

    return batch_data


def load_split_data(data_path: str, train_ratio: int = 0.85, test_ratio: int = 0.10):
    """
    data split
    """
    data = load_json_data(data_path = data_path)
    logger.info(f"Number of entries: {len(data)}")
    # data split ratio
    train_portion = int(len(data) * train_ratio)  # 85% 用作训练集
    test_portion = int(len(data) * test_ratio)    # 10% 用作测试集
    valid_portion = len(data) - train_portion - test_portion  # 剩下的 5% 用作验证集
    # data split
    train_data = data[:train_portion]
    test_data = data[train_portion:(train_portion + test_portion)]
    valid_data = data[(train_portion + test_portion):]
    # logger.info(f"Train data lenght: {len(train_data)}")
    # logger.info(f"Test data length: {len(test_data)}")
    # logger.info(f"Valid data length: {len(valid_data)}")

    return train_data, test_data, valid_data 


def create_dataloader(
        data, 
        device,
        tokenizer,
        pad_token_id = 50256,
        mask_prompt_tokens = True,
        allowed_max_length = 1024,
        batch_size: int = 2, 
        shuffle: bool = False, 
        drop_last: bool = False, 
        num_workers: int = 0
    ):
    # collate function
    collate_fn = partial(
        custom_collate_fn, 
        pad_token_id = pad_token_id,
        mask_prompt_tokens = mask_prompt_tokens, 
        allowed_max_length = allowed_max_length,
        device = device, 
    )
    # data set
    dataset = PreferenceDataset(data = data, tokenizer = tokenizer)
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
    from data_provider.finetune.dpo import data_config
    from utils.device import device

    # params
    batch_size = 2

    # tokenizer
    import tiktoken
    tokenizer = tiktoken.get_encoding("gpt2")

    # data load and split
    train_data, test_data, valid_data = load_split_data(
        data_path = data_config.instruction_data_with_preference_path,
        train_ratio = 0.85,
        test_ratio = 0.10,
    )
    # dataset and dataloader
    train_dataset, train_dataloader = create_dataloader(
        data = train_data,
        device = device,
        tokenizer = tokenizer,
        pad_token_id = 50256,
        mask_prompt_tokens =True,
        allowed_max_length = 1024,
        batch_size = batch_size,
        shuffle = True,
        drop_last = True,
        num_workers = 0,
    )
    # test_dataset, test_dataloader = create_dataloader(
    #     data = test_data,
    #     device = device,
    #     tokenizer = tokenizer,
    #     pad_token_id = 50256,
    #     mask_prompt_tokens =True,
    #     allowed_max_length = 1024,
    #     batch_size = batch_size,
    #     shuffle = False,
    #     drop_last = False,
    #     num_workers = 0,
    # )
    # valid_dataset, valid_dataloader = create_dataloader(
    #     data = valid_data,
    #     device = device,
    #     tokenizer = tokenizer,
    #     pad_token_id = 50256,
    #     mask_prompt_tokens =True,
    #     allowed_max_length = 1024,
    #     batch_size = batch_size,
    #     shuffle = False,
    #     drop_last = False,
    #     num_workers = 0, 
    # )
    # logger.info(f"Training data length: {len(train_data)} Training batches: {len(train_dataloader)}")
    # logger.info(f"Test data length: {len(test_data)} \tTest batches: {len(test_dataloader)}")
    # logger.info(f"Valid data length: {len(valid_data)} \tValidation batches: {len(valid_dataloader)}")

    # test
    logger.info(f"Train loader:")
    for batch in train_dataloader:
        # logger.info(f"chosen.shape: {batch['chosen'].shape}, rejected.shape: {batch['rejected'].shape}")
        pass
    # ------------------------------
    # test
    # ------------------------------
    # def decode_tokens_from_batch(token_ids):
    #     import tiktoken
    #     tokenizer = tiktoken.get_encoding("gpt2")
    #     ids_in_python_list = token_ids.flatten().tolist()

    #     return tokenizer.decode(ids_in_python_list)

    # text = decode_tokens_from_batch(batch["prompt"][0])
    # logger.info(f"prompt text: \n{text}")

    # text = decode_tokens_from_batch(batch["chosen"][0])
    # logger.info(f"chosen text: \n{text}")

    # text = decode_tokens_from_batch(batch["rejected"][0])
    # logger.info(f"rejected text: \n{text}")
    
    # text = decode_tokens_from_batch(token_ids=batch["chosen"][0][batch["chosen_mask"][0]])
    # logger.info(f"chosen mask text: \n{text}")

    # text = decode_tokens_from_batch(token_ids=batch["chosen"][0][batch["rejected_mask"][0]])
    # logger.info(f"rejected mask text: \n{text}")

if __name__ == "__main__":
    main()
