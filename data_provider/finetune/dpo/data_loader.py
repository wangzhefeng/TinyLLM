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

from data_provider.finetune import instruction_format
from utils.device import device
from utils.log_util import logger

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


class PreferenceDataset(Dataset):

    def __init__(self, data, tokenizer):
        # load data
        self.data = data
        # pre-tokenize texts
        self.encoded_texts = []
        for entry in self.data:
            # format input into instruction-response template
            prompt = instruction_format.format_input_alpaca(entry)
            rejected_response = entry["rejected"]
            chosen_response = entry["chosen"]
            # prompt tokenization
            prompt_tokens = tokenizer.encode(prompt)
            # response format and tokenization
            chosen_full_text = f"{prompt}\n\n### Response:\n{chosen_response}"
            rejected_full_text = f"{prompt}\n\n### Response:\n{rejected_response}"
            chosen_full_tokens = tokenizer.encode(chosen_full_text)
            rejected_full_tokens = tokenizer.encode(rejected_full_text)

            self.encoded_texts.append({
                "prompt": prompt_tokens,
                "chosen": chosen_full_tokens,
                "rejected": rejected_full_tokens,
            })

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
        "rejected_mask": [],
        "chosen_mask": []
    }
    # 确定最长的序列以设置相同的填充长度
    max_length_common = 0
    if batch:
        for key in ["chosen", "rejected"]:
            current_max = max(len(item[key]) + 1 for item in batch)
            max_length_common = max(max_length_common, current_max)

    # 处理批次中的每个项目
    for item in batch:
        # prompt
        prompt = torch.tensor(item["prompt"])
        batch_data["prompt"].append(prompt)
        # 
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


def create_dataloader(data, 
                      tokenizer,
                      pad_token_id = 50256,
                      mask_prompt_tokens = True,
                      allowed_max_length = 1024,
                      batch_size: int = 2, 
                      shuffle: bool = False, 
                      drop_last: bool = False, 
                      num_workers: int = 0):
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
    from data_provider.finetune.dpo import data_load

    # params
    batch_size = 2
    
    # data
    data = data_load.load_dpo_data(data_path = data_config.data_path)
    
    # example
    import pprint
    example_data = data[:2]
    for i in example_data:
        logger.info(f"data: \n{i}")
    
    # dataset and dataloader
    dataset, dataloader = create_dataloader(data[:2])
    for batch in dataloader:
        break
    logger.info(f"batch.keys: \n{batch.keys()}")
    logger.info(f"batch['prompt']: \n{batch['prompt']}")
    logger.info(f"batch['chosen']: \n{batch['chosen']}")

    # test
    def decode_tokens_from_batch(token_ids):
        import tiktoken
        tokenizer = tiktoken.get_encoding("gpt2")
        ids_in_python_list = token_ids.flatten().tolist()

        return tokenizer.decode(ids_in_python_list)

    text = decode_tokens_from_batch(batch["prompt"][0])
    logger.info(f"prompt text: \n{text}")

    text = decode_tokens_from_batch(batch["chosen"][0])
    logger.info(f"chosen text: \n{text}")

    text = decode_tokens_from_batch(batch["rejected"][0])
    logger.info(f"rejected text: \n{text}")
    
    text = decode_tokens_from_batch(token_ids=batch["chosen"][0][batch["chosen_mask"][0]])
    logger.info(f"chosen mask text: \n{text}")

    text = decode_tokens_from_batch(token_ids=batch["chosen"][0][batch["rejected_mask"][0]])
    logger.info(f"rejected mask text: \n{text}")

if __name__ == "__main__":
    main()
