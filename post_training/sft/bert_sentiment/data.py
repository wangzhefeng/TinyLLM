# -*- coding: utf-8 -*-

# ***************************************************
# * File        : data.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-10-18
# * Version     : 1.0.101800
# * Description : 中文情感分类数据管线：加载 sentiment 数据集，字符级编码（复用 THUCNews 词表）
# * Link        : docs/40-models/bert.md
# * Requirement : torch
# ***************************************************

# python libraries
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)
import warnings
warnings.filterwarnings("ignore")

import torch
from torch.utils.data import Dataset

from utils.log_util import logger
from tools.data.io import read_text_label_tsv

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


def load_sentiment_file(file_path: str):
    """
    加载情感分类数据文件（每行：text<TAB>label），返回 (texts, labels)。
    解析逻辑统一在 tools.data.io.read_text_label_tsv。
    """
    texts, labels = read_text_label_tsv(file_path)
    logger.info(f"Loaded {len(texts)} samples from {file_path}")
    return texts, labels


def load_sentiment_data(data_dir: str, max_samples: int = -1):
    """
    加载 train / valid / test 三个 split
    """
    data_dir = Path(data_dir)
    splits = {}
    for name in ["train", "valid", "test"]:
        texts, labels = load_sentiment_file(str(data_dir / f"sentiment.{name}.data"))
        if name == "train" and max_samples and max_samples > 0:
            texts, labels = texts[:max_samples], labels[:max_samples]
            logger.info(f"Train split truncated to {len(texts)} samples")
        splits[name] = (texts, labels)
    return splits["train"], splits["valid"], splits["test"]


class SentimentDataset(Dataset):
    """
    单句分类输入：[CLS] text [SEP] + [PAD] 填充；segment 全 0（padding 为 2，与预训练一致）。
    字符级编码，未登录字映射 [UNK]。
    """

    def __init__(self, texts, labels, word_2_index, max_len: int = 128):
        self.texts = texts
        self.labels = labels
        self.word_2_index = word_2_index
        self.max_len = max_len

    def __getitem__(self, index):
        w2i = self.word_2_index
        unk_idx = w2i["[UNK]"]
        text_idx = [w2i.get(ch, unk_idx) for ch in self.texts[index]][: self.max_len - 2]
        input_ids = [w2i["[CLS]"]] + text_idx + [w2i["[SEP]"]]
        pad_len = self.max_len - len(input_ids)
        seg_ids = [0] * len(input_ids) + [2] * pad_len
        input_ids = input_ids + [w2i["[PAD]"]] * pad_len
        return (
            torch.tensor(input_ids),
            torch.tensor(seg_ids),
            torch.tensor(self.labels[index]),
        )

    def __len__(self):
        return len(self.labels)




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
