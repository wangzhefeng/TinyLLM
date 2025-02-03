# -*- coding: utf-8 -*-

# ***************************************************
# * File        : tokenizer.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-01-23
# * Version     : 1.0.012322
# * Description : BPE(BytePair tokenizer): https://github.com/openai/gpt-2/blob/master/src/encoder.py
# *                                        https://github.com/rasbt/LLMs-from-scratch/blob/0911e71497769782975d68dba6e13f22157e5fb5/ch02/02_bonus_bytepair-encoder/compare-bpe-tiktoken.ipynb
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# * TODO        : 1.
# ***************************************************

__all__ = []

# python libraries
import os
import sys
ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
import re
from typing import List, Dict

from utils.log_util import logger

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


def build_vocab(text: str):
    """
    Converting tokens into token IDs

    Args:
        tokenization_set (List): _description_
    """
    logger.info("Build Vocab: Converting tokens into token IDs...")
    # 训练数据分词
    token_list = re.split(r'([,.:;?_!"()\']|--|\s)', text)
    token_list = [item.strip() for item in token_list if item.strip()]
    # 训练数据所有 token(不重复)
    all_tokens = sorted(set(token_list))
    all_tokens.extend(["<|endoftext|>", "<|unk|>"])
    logger.info(f"Vocab size: {len(all_tokens)}")
    # 构建词典
    vocab = {
        token: integer
        for integer, token in enumerate(all_tokens)
    }
    
    return vocab


class SimpleTokenizer:

    def __init__(self, vocab: Dict):
        self.str_to_int = vocab
        self.int_to_str = {
            i: s 
            for s, i in vocab.items()
        }

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

    def decode(self, token_ids: List):
        """
        token IDs decode to text
        """
        text = " ".join([self.int_to_str[i] for i in token_ids])
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
    input_text_2 = """It's the last he painted, you know," 
                    Mrs. Gisburn said with pardonable pride."""
    input_text_3 = "Hello, do you like tea. Is this-- a test?"

    # 训练数据下载、加载
    from tiny_model.TinyLLM.data_load_pretrain import data_download, data_load
    file_path = data_download()
    raw_text = data_load(file_path=file_path)

    # 构建数据词典
    from tiny_model.TinyLLM.tokenization import build_vocab, SimpleTokenizer
    vocab = build_vocab(text=raw_text)
    logger.info(f"vocab: {vocab}")

    # method 1: simple tokenizer
    tokenizer = SimpleTokenizer(vocab=vocab)
    token_ids = tokenizer.encode(text=input_text_1)
    logger.info(f"token_ids: {token_ids}")
    decoded_text = tokenizer.decode(token_ids=token_ids)
    logger.info(f"decoded_text: {decoded_text}") 
    
    # method 2: BPE: tiktoken
    import tiktoken
    tokenizer = tiktoken.get_encoding("gpt2")
    token_ids = tokenizer.encode(text=input_text_1, allowed_special={"<|endoftext|>"})
    logger.info(f"token_ids: {token_ids}")
    decoded_text = tokenizer.decode(tokens=token_ids)
    logger.info(f"decoded_text: {decoded_text}")

if __name__ == "__main__":
    main()
