# -*- coding: utf-8 -*-

# ***************************************************
# * File        : data_loader.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-05-03
# * Version     : 1.0.050301
# * Description : description
# * Link        : https://github.com/jingyaogong/minimind/blob/master/dataset/lm_dataset.py
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# * TODO        : 1.
# ***************************************************

__all__ = []

# python libraries
import os
import sys
ROOT = str(os.getcwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)
import re
import ast
import json
import random

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader

from utils.log_util import logger

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


class PretrainDataset(Dataset):
    
    def __init__(self, data_path, tokenizer, max_length: int=512):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = self.load_data(data_path)
    
    def load_data(self, path):
        samples = []
        with open(path, "r", encoding="utf-8") as file:
            for line_num, line in enumerate(file, start=1):
                data = json.loads(line.strip())
                logger.info(f"debug::data: \n{data}")
                samples.append(data)
                logger.info(f"debug::samples: \n{samples}")
                if line_num == 2:
                    break
        
        return samples

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        sample = self.samples[index]
        # 构建输入文本
        encoding = self.tokenizer(
            str(sample["text"]),
            max_length = self.max_length,
            padding = "max_length",
            truncation = True,
            return_tensors = "pt"
        )
        input_ids = encoding.input_ids.squeeze()
        loss_mask = (input_ids != self.tokenizer.pad_token_id)

        X = torch.tensor(input_ids[:-1], dtype=torch.long)
        Y = torch.tensor(input_ids[1:], dtype=torch.long)
        loss_mask = torch.tensor(loss_mask[1:], dtype=torch.long)

        return X, Y, loss_mask


class SFTDataset(Dataset):
    
    def __init__(self):
        super().__init__()


class DPODataset(Dataset):
    
    def __init__(self):
        super().__init__()


class RLAIFDataset(Dataset):
    
    def __init__(self):
        super().__init__()



# 测试代码 main 函数
def main():
    pretrain_dataset = PretrainDataset(
        data_path="./minimind/dataset/pretrain_hq.jsonl",
        tokenizer=None,
        max_length=512,
    )

if __name__ == "__main__":
    main()
