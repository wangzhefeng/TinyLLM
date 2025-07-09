# -*- coding: utf-8 -*-

# ***************************************************
# * File        : tokenizer.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-01-31
# * Version     : 1.0.013122
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
from typing import List


from sentencepiece import SentencePieceProcessor

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


TOKENIZER_MODEL = "tokenizer.model"


class Tokenizer:
    
    def __init__(self, tokenizer_model):
        model_path = tokenizer_model if tokenizer_model else TOKENIZER_MODEL
        self.sp_model = SentencePieceProcessor(model_file=model_path)
        self.model_path = model_path

        self.n_words = self.sp_model.vocab_size()
        self.bos_id = self.sp_model.bos_id()
        self.eos_id = self.sp_model.eos_id()
        self.pad_id = self.sp_model.pad_id()

    def encode(self, s: str, bos: bool, eos: bool) -> List[int]:
        t = self.sp_model.encode(s)
        if bos:
            t = [self.bos_id] + t
        if eos:
            t = t + [self.eos_id]
        return t

    def decode(self, tokens: List[int]) -> str:
        return self.sp_model.decode(tokens)




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
