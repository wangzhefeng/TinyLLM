# -*- coding: utf-8 -*-

# ***************************************************
# * File        : tokenizer.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-01-01
# * Version     : 1.0.010121
# * Description : BERT 采用了 WordPiece 分词，根据训练语料中的词频决定是否将一个完整的词切分为多个词元。因此，需要首先训练词元分析器（Tokenizer）
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

__all__ = []

# python libraries
import os
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)
import json

from layers.tokenizers import BertWordPieceTokenizer

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


def tokenizer_train(train_data, vocab_size, special_tokens, max_len, model_path):
    # 初始化 WordPiece 词元分析器
    tokenizer = BertWordPieceTokenizer()
    # 训练词元分析器
    tokenizer.train(files=train_data, vocab_size=vocab_size, special_tokens=special_tokens)
    # 允许截断达到最大 512 词元
    tokenizer.enable_truncation(max_len=max_len)
    # 保存词元分析器模型
    if not os.path.isdir(model_path):
        os.mkdir(model_path)
    tokenizer.save_model(model_path)
    # 将一些词元分析器中的配置保存到配置文件，包括特殊词元，转换为小写，最大序列长度等
    with open(os.path.join(model_path, "config.json"), "w", encoding="utf-8") as f:
        tokenizer_cfg = {
            "do_lower_case": True,
            "unk_token": "[UNK]",
            "sep_token": "[SEP]",
            "pad_token": "[PAD]",
            "cls_token": "[CLS]",
            "mask_token": "[MASK]",
            "model_max_length": max_len,
            "max_len": max_len,
        }
        json.dump(tokenizer_cfg, f)





# 测试代码 main 函数
def main():
    data_dir = "/Users/wangzf/llm_proj/dataset/bert_data/"
    project_dir = "/Users/wangzf/llm_proj/tiny_llm//bert/"
    model_path = os.path.join(project_dir, "pretrained-bert")
    train_data = [os.path.join(data_dir, "train.txt")]  # 仅根据训练集合训练次元分析器
    # BERT 中采用的默认词表大小为 30522，可以根据需要调整
    vocab_size = 30522
    # 特殊字符
    special_tokens = ["[PAD]", "UNK", "[CLS]", "[SEP]", "[MASK]", "<S>", "T"]
    # 最大序列长度，长度越低训练速度越快
    max_len = 512
    
    # 模型训练
    tokenizer_train(train_data, vocab_size, special_tokens, max_len, model_path)


if __name__ == "__main__":
    main()
