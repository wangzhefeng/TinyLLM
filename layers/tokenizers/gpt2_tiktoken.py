# -*- coding: utf-8 -*-

# ***************************************************
# * File        : gpt2_tiktoken.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-03-30
# * Version     : 1.0.033016
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


import tiktoken

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


class GPT2Tokenizer:
    
    def __init__(self, tokenizer_model: str = "gpt2"):
        self.tokenizer = tiktoken.get_encoding(tokenizer_model)
    
    @property
    def n_vocab(self):
        return self.tokenizer.n_vocab

    def encode(self, text):
        return self.tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    
    def decode(self, ids):
        return self.tokenizer.decode(ids) 




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
 
    # method 2: BPE: tiktoken
    # ---------------------------------------
    tokenizer = GPT2Tokenizer()
    logger.info(f"tokenizer.n_vocab: {tokenizer.n_vocab}")
    
    token_ids = tokenizer.encode(text=input_text_1)
    logger.info(f"token_ids: {token_ids}")
    
    decoded_text = tokenizer.decode(token_ids)
    logger.info(f"decoded_text: {decoded_text}")

if __name__ == "__main__":
    main()
