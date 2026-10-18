# -*- coding: utf-8 -*-

# ***************************************************
# * File        : hf_gpt2_bpe.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-08-03
# * Version     : 1.0.080317
# * Description : description
# * Link        : https://huggingface.co/docs/transformers/model_doc/gpt2#transformers.GPT2Tokenizer
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

__all__ = []

# python libraries
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)
from typing import List
import warnings
warnings.filterwarnings("ignore")

from transformers import GPT2Tokenizer, GPT2TokenizerFast

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


class HuffingFaceGPT2BPE:

    def __init__(self):
        self.tokenizer = GPT2Tokenizer.from_pretrained(
            vocab_file="gpt2",  # vocabulary 文件路径
            merges_file="",  # 合并文件路径
            errors="replace",  # 解码字节为 UTF-8 时遵循的范式
            unk_token="<|endoftext|>",  # 未知标记
            bos_token="<|endoftext|>",  # 序列开始的标记
            eos_token="<|endoftext|>",  # 序列结束的标记
            pad_token="<|unk|>",  # 用于填充的标记，例如当批处理不同长度的序列时
            add_prefix_space=False,  # 是否在输入前添加一个空格。这允许将开头的单词与其他单词一样处理
            add_bos_token=False,  # 是否在输入中添加一个初始的句子开头标记。这允许将开头的词视为任何其他词
        )
        self.tokenizer_fast = GPT2TokenizerFast.from_pretrained("gpt2")
    
    @property
    def n_vocab(self):
        return self.tokenizer.total_vocab_size()

    def encode(self, text: str, fast: bool=True):
        """
        text encode to token IDs
        """
        if fast:
            token_ids = self.tokenizer_fast(text=text)["input_ids"]
        else:
            token_ids = self.tokenizer(text=text)["input_ids"]

        return token_ids
    
    def decode(self, tokens: List, fast: bool=True):
        """
        token IDs decode to text
        """
        if fast:
            text = self.tokenizer_fast(tokens=tokens)
        else:
            text = self.tokenizer(tokens=tokens)
        
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

    # BPE: huggingface
    tokenizer = HuffingFaceGPT2BPE()
    logger.info(f"tokenizer.n_vocab: {tokenizer.n_vocab}")

    token_ids = tokenizer.encode(text=input_text_1)
    logger.info(f"token_ids: {token_ids}")
    
    decoded_text = tokenizer.decode(tokens=token_ids)
    logger.info(f"decoded_text: {decoded_text}")

if __name__ == "__main__":
    main()
