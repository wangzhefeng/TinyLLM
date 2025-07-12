# -*- coding: utf-8 -*-

# ***************************************************
# * File        : tokenizer.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-01-23
# * Version     : 1.0.012322
# * Description : tokenizer
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


import torch

from tokenizers.simple_custom import SimpleTokenizer
from tokenizers.simple_bpe import BPETokenizerSimple
from tokenizers.gpt2_tiktoken import GPT2Tokenizer
from tokenizers.llama2_7b_sentencepiece import Llama27bTokenizer
from tokenizers.llama3_8b_bpe import Llama38bTokenizer

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


def choose_tokenizer(tokenizer_model: str = "gpt2"):
    """
    choose tokenizer
    """
    tokenizer_models_map = {
        "simple_custom": SimpleTokenizer,
        # "simple_bpe": BPETokenizerSimple,
        "gpt2": GPT2Tokenizer,
        "llama2": Llama27bTokenizer,
        "llama3-8B": Llama38bTokenizer,
    }
    tokenizer = tokenizer_models_map[tokenizer_model]()

    return tokenizer


def text_to_token_ids(text: str, tokenizer_model: str = "gpt2"):
    """
    tokenizer text to token_ids 
    """
    # tokenizer
    if isinstance(tokenizer_model, str):
        tokenizer = choose_tokenizer(tokenizer_model=tokenizer_model)
    else:
        tokenizer = tokenizer_model
    # text encode to token ids
    encoded = tokenizer.encode(text)
    # add batch dimension
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    
    return encoded_tensor


def token_ids_to_text(token_ids: List, tokenizer_model: str = "gpt2"):
    """
    tokenizer decoded token_ids to text 
    """
    # tokenizer
    if isinstance(tokenizer_model, str):
        tokenizer = choose_tokenizer(tokenizer_model=tokenizer_model)
    else:
        tokenizer = tokenizer_model
    # remove batch dimension
    token_ids_flat_list = token_ids.squeeze(0).tolist()
    # token ids decode to text
    decoded_text = tokenizer.decode(token_ids_flat_list)
    
    return decoded_text




# 测试代码 main 函数
def main():
    from utils.log_util import logger

    # input text
    input_text = (
        "Hello, do you like tea? <|endoftext|> In the sunlit terraces"
        "of someunknownPlace."
    )

    # test tokenizer
    tokenizer_model_name = "simple_bpe"
    token_ids = text_to_token_ids(input_text, tokenizer_model=tokenizer_model_name)
    decoded_text = token_ids_to_text(token_ids, tokenizer_model=tokenizer_model_name)
    logger.info(f"input_text: {input_text}")
    logger.info(f"decoded_text: {decoded_text}")
    
if __name__ == "__main__":
    main()
