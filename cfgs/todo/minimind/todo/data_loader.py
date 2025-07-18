# -*- coding: utf-8 -*-

# ***************************************************
# * File        : dataset.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-02-09
# * Version     : 0.1.020920
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
import re
import json
import random


import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


class PretrainDataset(Dataset):
    
    def __init__(self, df, tokenizer, max_length: int = 512):
        super().__init__()

        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = 0
    
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, index: int):
        # data sample
        sample = self.df.iloc[index]
        # text
        text = f"{self.tokenizer.bos_token}{str(sample["text"])}{self.tokenizer.eos_token}"
        # tokenization: text -> token ids
        input_id = self.tokenizer(text).data["input_ids"][:self.max_length]
        text_len = len(input_id)
        # 没满最大长度的剩余部分
        padding_len = self.max_length - text_len
        input_id = input_id + [self.padding] * padding_len
        # 0 表示不计算损失
        loss_mask = [1] * text_len + [0] * padding_len
        input_id = np.array(input_id)
        # convert to tensor
        X = np.array(input_id[:-1]).astype(np.int64)
        Y = np.array(input_id[1:]).astype(np.int64)
        loss_mask = np.array(loss_mask[1:]).astype(np.int64)

        return torch.from_numpy(X), torch.from_numpy(Y), torch.from_numpy(loss_mask)


class SFTDataset(Dataset):
    
    def __init__(self, df, tokenizer, max_length = 1024, prompt_max_len = 512, answer_max_len = 256):
        super().__init__()

        self.df = df
        self.max_length = max_length
        self.prompt_max_len = prompt_max_len
        self.answer_max_len = answer_max_len

        self.tokenizer = tokenizer
        self.padding = 0
        self.bos_id = self.tokenizer("<s>assistant").data["input_ids"]

    def __len__(self):
        return self.df.shape[0]
    
    def find_sublist_index(self, main_list, sub_list) -> int:
        last_index = -1
        for i in range(len(main_list) - len(sub_list) + 1):
            if main_list[i:i+len(sub_list)] == sub_list:
                last_index = i
        
        return last_index
    
    def safe_eval(self, s):
        try:
            res = eval(s)
        except Exception as e:
            return []
        
        return res
    
    def __getitem__(self, index: int):
        sample = self.df.iloc[index]
        # history, q, a
        history = self.safe_eval(sample["history"])
        q = str(sample["q"])
        a = str(sample["a"])
        # TODO
        messages = []
        for history_message in history:
            if len(history_message) <= 1:
                continue
            messages.append({"role": "user", "content": str(history_message[0])[:self.max_length // 2]})
            messages.append({"role": "assistant", "content": str(history_message[1])[:self.max_length // 2]})
        messages += [
            {"role": "user", "content": q},
            {"role": "assistant", "content": a},
        ]
        new_prompt = self.tokenizer.apply_chat_template(
            messages, 
            tokenize = False, 
            add_generation_prompt = True
        )
        input_id = self.tokenizer(new_prompt).data["input_ids"][:self.max_length]
        # 实际长度
        question_length = self.find_sublist_index(input_id, self.bos_id) + len(self.bos_id)
        # 没满最大长度的剩余部分
        padding_len = self.max_length - len(input_id)
        input_id = input_id + [self.padding] * padding_len
        mask_len = len(input_id) - question_length - padding_len
        # 0 表示不计算损失
        loss_mask = [0] * question_length + [1] * (mask_len) + [0] * padding_len

        input_id = np.array(input_id)
        X = np.array(input_id[:-1]).astype(np.int64)
        Y = np.array(input_id[1:]).astype(np.int64)
        loss_mask = np.array(loss_mask[1:]).astype(np.int64)

        X_tensor = torch.from_numpy(X)
        Y_tensor = torch.from_numpy(Y)
        loss_mask_tensor = torch.from_numpy(loss_mask)

        return X_tensor, Y_tensor, loss_mask_tensor



# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
