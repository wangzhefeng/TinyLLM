# -*- coding: utf-8 -*-

# ***************************************************
# * File        : tiktoken_bpe.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-08-01
# * Version     : 1.0.080123
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

__all__ = []

# python libraries
from pathlib import Path
from typing import List
import warnings
warnings.filterwarnings("ignore")

import tiktoken

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


class TiktokenBPE:

    def __init__(self, tokenizer_model: str = "gpt2"):
        self.tokenizer = tiktoken.get_encoding(tokenizer_model)
        # logger.info(f"tokenizer.n_vocab: {self.tokenizer.n_vocab}")
        # logger.info(f"tokenizer.eot_token: {self.tokenizer.eot_token}")
    
    @property
    def n_vocab(self):
        return self.tokenizer.n_vocab

    def encode(self, text: str, allowed_special=None, **kwargs):
        """
        text encode to token IDs

        Args:
            allowed_special: 允许的特殊 token 集合（默认 {"<|endoftext|>"}）；
                透传 tiktoken 的同名参数，调用方可显式传入
        """
        if allowed_special is None:
            allowed_special = {"<|endoftext|>"}
        token_ids = self.tokenizer.encode(text=text, allowed_special=allowed_special, **kwargs)

        return token_ids
    
    def decode(self, tokens: List):
        """
        token IDs decode to text
        """
        text = self.tokenizer.decode(tokens=tokens)
        
        return text




# 测试代码 main 函数
def main():
    from utils.log_util import logger
    
    # input text
    input_text_1 = (
        "Hello, do you like tea? <|endoftext|> In the sunlit terraces"
        "of someunknownPlace."
    )
    input_text_2 = "Hello, do you like tea? <|endoftext|> In the sunlit terraces of someunknownPlace."
    input_text_3 = """It's the last he painted, you know," 
                    Mrs. Gisburn said with pardonable pride."""
    input_text_4 = "Hello, do you like tea. Is this-- a test?"

    # corpus
    from utils.load_save_data import load_local_data
    raw_text = load_local_data(
        url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt",
        data_path = "./dataset/pretraining",
        data_file = "the-verdict.txt",
    )

    # BPE: tiktoken
    tokenizer = TiktokenBPE() 

    corpus_token_ids = tokenizer.encode(text=raw_text)
    logger.info(f"corpus_token_ids: {len(corpus_token_ids)}")

    token_ids = tokenizer.encode(text=input_text_1)
    logger.info(f"token_ids: {token_ids}")
    
    decoded_text = tokenizer.decode(tokens=token_ids)
    logger.info(f"decoded_text: {decoded_text}")

if __name__ == "__main__":
    main()
