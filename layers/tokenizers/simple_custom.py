# -*- coding: utf-8 -*-

# ***************************************************
# * File        : simple_tokenizer.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-03-30
# * Version     : 1.0.033003
# * Description : description
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
import re
from typing import List


from data_provider.pretrain.data_loader import data_load
from utils.log_util import logger

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


class SimpleTokenizer:

    def __init__(self, raw_text: str = None):
        raw_text = data_load(
            # url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt",
            data_path = "./dataset/pretrain/gpt",
            data_file = "the-verdict.txt",
        )
        self.vocab = self._build_vocab(raw_text)
        self.str_to_int = self.vocab
        self.int_to_str = {i: s for s, i in self.vocab.items()}

    def _build_vocab(self, text: str):
        """
        Build vocab
        Converting tokens into token IDs
        """
        # logger.info("Build Vocab: Converting tokens into token IDs...")
        # 训练数据分词
        token_list = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        token_list = [item.strip() for item in token_list if item.strip()]
        # self.n_token = len(token_list)
        # logger.info(f"Token size: {self.n_token}")
        # 训练数据所有 token(不重复)
        self.all_tokens = sorted(set(token_list))
        # special tokens: [BOS], [EOS], [PAD], [UNK], [endoftext], <UNK>
        self.all_tokens.extend(["<|endoftext|>", "<|unk|>"]) 
        # 构建词典
        vocab = {
            token: integer
            for integer, token in enumerate(self.all_tokens)
        }
        
        return vocab
    
    @property
    def n_vocab(self):
        return len(self.all_tokens)

    def encode(self, text: str):
        """
        text encode to token IDs
        """
        tokens = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        tokens = [item.strip() for item in tokens if item.strip()]
        tokens = [
            item
            if item in self.str_to_int else "<|unk|>"
            for item in tokens
        ]
        token_ids = [self.str_to_int[s] for s in tokens]

        return token_ids

    def decode(self, tokens: List):
        """
        token IDs decode to text
        """
        text = " ".join([self.int_to_str[i] for i in tokens])
        # Replace spaces before the specified punctuations
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        
        return text




# 测试代码 main 函数
def main():
    # input text
    input_text_1 = (
        "Hello, do you like tea? <|endoftext|> In the sunlit terraces"
        "of someunknownPlace."
    )
    input_text_2 = "Hello, do you like tea? <|endoftext|> In the sunlit terraces of someunknownPlace."
    input_text_3 = """It's the last he painted, you know," 
                    Mrs. Gisburn said with pardonable pride."""
    input_text_4 = "Hello, do you like tea. Is this-- a test?"

    # simple tokenizer
    tokenizer = SimpleTokenizer()
    logger.info(f"tokenizer.n_vocab: {tokenizer.n_vocab}")
    
    token_ids = tokenizer.encode(text=input_text_2)
    logger.info(f"token_ids: {token_ids}")
    
    decoded_text = tokenizer.decode(tokens=token_ids)
    logger.info(f"decoded_text: {decoded_text}")

if __name__ == "__main__":
    main()
