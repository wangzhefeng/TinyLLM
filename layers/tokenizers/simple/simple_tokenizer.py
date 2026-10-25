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
from pathlib import Path
import re
from typing import List

from utils.log_util import logger

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


class SimpleTokenizer:
    # TokenizerProtocol 属性：simple tokenizer 无 eos/pad 特殊 token id
    eos_token_id = None
    pad_token_id = None

    def __init__(self, text: str=None):
        vocab = self._build_vocab(text)
        # for token, token_id in vocab.items():
        #     logger.info(f"(token: token_id) ({token}: {token_id})")
        #     if token_id >= 50:
        #         break
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}

    def _build_vocab(self, text: str):
        """
        Build vocabulary: Converting text into {tokens, token IDs}
        """
        # logger.info("Build Vocab: Converting tokens into token IDs...")
        # 训练数据分词
        token_list = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        token_list = [item.strip() for item in token_list if item.strip()]
        # 训练数据所有 token(不重复)
        self.all_tokens = sorted(set(token_list))
        # special tokens: [BOS], [EOS], [PAD], [UNK], [endoftext], <UNK>
        self.all_tokens.extend(["<|endoftext|>", "<|unk|>"])
        # 构建词典
        vocab = {
            token: token_id
            for token_id, token in enumerate(self.all_tokens)
        }
        
        return vocab
    
    @property
    def n_vocab(self):
        return len(self.all_tokens)

    def encode(self, text: str):
        """
        text encode to token IDs
        """
        # 输入数据分词
        tokens = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        tokens = [item.strip() for item in tokens if item.strip()]
        # 输入数据 token 处理(加入未知 token)
        tokens = [
            item if item in self.str_to_int else "<|unk|>"
            for item in tokens
        ]
        # token 转换为 token ID
        token_ids = [self.str_to_int[s] for s in tokens]

        return token_ids

    def decode(self, tokens: List):
        """
        token IDs decode to text
        """
        # token ID 转换为 token
        text = " ".join([self.int_to_str[i] for i in tokens])
        # Replace spaces before the specified punctuations
        text = re.sub(r'\s+([,.:;?!"()\'])', r'\1', text)
        
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

    # corpus
    from utils.load_save_data import load_local_data
    raw_text = load_local_data(
        url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt",
        data_path = "./dataset/pretraining",
        data_file = "the-verdict.txt",
    )

    # simple tokenizer
    tokenizer = SimpleTokenizer(text=raw_text)
    logger.info(f"tokenizer.n_vocab: {tokenizer.n_vocab}")
    
    token_ids = tokenizer.encode(text=input_text_2)
    logger.info(f"token_ids: {token_ids}")
    
    decoded_text = tokenizer.decode(tokens=token_ids)
    logger.info(f"decoded_text: {decoded_text}")

if __name__ == "__main__":
    main()
