# -*- coding: utf-8 -*-

# ***************************************************
# * File        : data_provider.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-10-15
# * Version     : 1.0.101523
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
import random
import warnings
warnings.filterwarnings("ignore")

import torch
from torch.utils.data import Dataset

from utils.log_util import logger

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


class BertDataset(Dataset):
    def __init__(self, all_text1, all_text2, all_label, max_len, word_2_index):
        self.all_text1 = all_text1
        self.all_text2 = all_text2
        self.all_label = all_label
        self.max_len = max_len
        self.word_2_index = word_2_index

    def __getitem__(self, index):
        # 数据索引
        # ------------------------------
        text1 = self.all_text1[index]
        text2 = self.all_text2[index]
        lable = self.all_label[index]
        # 未知词替换
        # ------------------------------
        unk_idx = self.word_2_index["[UNK]"]
        text1_idx = [self.word_2_index.get(i, unk_idx) for i in text1][:62]
        text2_idx = [self.word_2_index.get(i, unk_idx) for i in text2][:63]
        # 序列阶段和填充
        # ------------------------------
        # 验证 text1_idx 中的索引是否越界，如果越界，替换为 [UNK] 索引
        for i, idx in enumerate(text1_idx):
            if idx >= len(self.word_2_index):
                logger.info(f"Index out of range in text1 at position {i}: {idx}, replacing with [UNK]")
                text1_idx[i] = unk_idx  # 替换为 [UNK] 的索引
        # 验证 text2_idx 中的索引是否越界，如果越界，替换为 [UNK] 索引
        for i, idx in enumerate(text2_idx):
            if idx >= len(self.word_2_index):
                logger.info(f"Index out of range in text2 at position {i}: {idx}, replacing with [UNK]")
                text2_idx[i] = unk_idx  # 替换为 [UNK] 的索引
        # 掩码操作
        # ------------------------------
        mask_val = [0] * self.max_len
        # 构建段索引和序列索引：用于区分句子对中的句子 A 和句子 B。同时，将特殊标记（如[CLS]和[SEP]）纳入序列，以符合BERT模型的输入格式
        text_idx = [self.word_2_index["[CLS]"]] + text1_idx + [self.word_2_index["[SEP]"]] + text2_idx + [self.word_2_index["[SEP]"]]
        seg_idx  = [0] + [0] * len(text1_idx) + [0] + [1] * len(text2_idx) + [1] + [2] * (self.max_len - len(text_idx))
        # 掩码
        for i, v in enumerate(text_idx):
            if v in [self.word_2_index["[CLS]"], self.word_2_index["[SEP]"], self.word_2_index["[UNK]"]]:
                continue
            if random.random() < 0.15:
                r = random.random()
                if r < 0.8:
                    text_idx[i] = self.word_2_index["[MASK]"]
                    mask_val[i] = v
                elif r > 0.9:
                    other_idx = random.randint(6, len(self.word_2_index) - 1)
                    text_idx[i] = other_idx
                    mask_val[i] = v
        text_idx = text_idx + [self.word_2_index["[PAD]"]] * (self.max_len - len(text_idx))

        return torch.tensor(text_idx), torch.tensor(lable), torch.tensor(mask_val), torch.tensor(seg_idx)

    def __len__(self):
        return len(self.all_label)




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
