# -*- coding: utf-8 -*-

# ***************************************************
# * File        : chat_format.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-04-05
# * Version     : 1.0.040500
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


# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


class ChatFormat:
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        
    def encode_header(self, message):
        tokens = []
        tokens.append(self.tokenizer.special_tokens["<|start_header_id|>"])
        tokens.extend(self.tokenizer.encode(message["role"], bos = False, eos = False))
        tokens.append(self.tokenizer.special_tokens["<|end_header_id|>"])
        tokens.extend(self.tokenizer.encode("\n\n", bos = False, eos = False))

        return tokens
    
    def encode(self, text):
        message = {
            "role": "user",
            "content": text,
        }
        tokens = self.encode_header(message)
        tokens.extend(self.tokenizer.encode(message["content"].strip(), bos = False, eos = False))
        tokens.append(self.tokenizer.special_tokens["<|eot_id|>"])

        return tokens
    
    def decode(self, token_ids):
        return self.tokenizer.decode(token_ids)




# 测试代码 main 函数
def main():
    from utils.log_util import logger

    # tokenizer
    from tokenizers.llama3_8b_bpe import Llama38bTokenizer
    tokenizer = Llama38bTokenizer(special_token_version="v1")

    # chat tokenizer
    chat_tokenizer = ChatFormat(tokenizer)

    # encode
    token_ids = chat_tokenizer.encode("Hello World!")
    logger.info(f"token_ids: {token_ids}")

    # deocde
    decoded_text = chat_tokenizer.decode(token_ids)
    logger.info(f"decoded_text: {decoded_text}")

if __name__ == "__main__":
    main()
