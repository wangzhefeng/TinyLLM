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
# * TODO        : 1.
# ***************************************************

__all__ = []

# python libraries
import os
import sys
ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
from typing import List

import torch
import tiktoken

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


def choose_tokenizer(tokenizer_model: str = "gpt2"):
    """
    choose tokenizer
    """
    tokenizer = tiktoken.get_encoding(tokenizer_model)

    return tokenizer


def text_to_token_ids(text: str, tokenizer_model: str = "gpt2"):
    """
    tokenizer text to token_ids

    Args:
        text (str): _description_

    Returns:
        _type_: _description_
    """
    # tokenizer
    tokenizer = choose_tokenizer(tokenizer_model=tokenizer_model)
    # text encode to token ids
    encoded = tokenizer.encode(text, allowed_special = {"<|endoftext|>"})
    # add batch dimension
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    
    return encoded_tensor


def token_ids_to_text(token_ids: List, tokenizer_model: str = "gpt2"):
    """
    tokenizer decoded token_ids to text

    Args:
        token_ids (_type_): _description_

    Returns:
        _type_: _description_
    """
    # tokenizer
    tokenizer = choose_tokenizer(tokenizer_model=tokenizer_model)
    tokenizer = tiktoken.get_encoding("gpt2")
    # remove batch dimension
    flat = token_ids.squeeze(0)
    # token ids decode to text
    decoded_text = tokenizer.decode(flat.tolist())
    
    return decoded_text




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
