# -*- coding: utf-8 -*-

# ***************************************************
# * File        : gemma.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-09-15
# * Version     : 1.0.091520
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
import warnings
warnings.filterwarnings("ignore")

from tokenizers import Tokenizer
from huggingface_hub import hf_hub_download

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]
os.environ['LOG_NAME'] = LOGGING_LABEL
from utils.log_util import logger


def tokenizer_download(repo_id: str, local_dir: str):
    tokenizer_file_path = Path(local_dir).joinpath("tokenizer.json")
    
    if not tokenizer_file_path.exists():
        try:
            tokenizer_file_path = hf_hub_download(
                repo_id=repo_id,
                filename="tokenizer.json",
                local_dir=local_dir,
            )
        except Exception as e:
            logger.info(f"Warning: Failed to download tokenizer.json: {e}")
            tokenizer_file_path = "tokenizer.json"
    
    return tokenizer_file_path


class Gemma3Tokenizer:
    
    def __init__(self, tokenizer_file_path: str):
        tok_file = Path(tokenizer_file_path)
        self._tok = Tokenizer.from_file(str(tok_file))
        # Attempt to identify EOS and padding tokens
        eos_token = "<end_of_turn>"
        self.pad_token_id = eos_token
        self.eos_token_id = eos_token

    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text).ids

    def decode(self, ids: list[int]) -> str:
        return self._tok.decode(ids, skip_special_tokens=False)


def apply_chat_template(user_text):
    return f"<start_of_turn>user\n{user_text}<end_of_turn>\n<start_of_turn>model\n"




# 测试代码 main 函数
def main():
    USE_INSTRUCT_MODEL = False
    CHOOSE_MODEL = "270m"
    
    if USE_INSTRUCT_MODEL:
        repo_id = f"google/gemma-3-{CHOOSE_MODEL}-it"
    else:
        repo_id = f"google/gemma-3-{CHOOSE_MODEL}"
    
    # tokenizer model dir
    tokenizer_model_dir = Path("./layers/tokenizers/models/").joinpath(Path(repo_id).parts[-1])

    # tokenizer model download
    tokenizer_model_file = tokenizer_download(repo_id=repo_id, local_dir=tokenizer_model_dir)

    # tokenizer model test
    tokenizer = Gemma3Tokenizer(tokenizer_model_file)

    # prompt
    prompt = "Give me a short introduction to large language models."
    prompt = apply_chat_template(prompt)
    logger.info(f"prompt: {prompt}")

    input_token_ids = tokenizer.encode(prompt)
    logger.info(f"input_token_ids: {input_token_ids}")

    text = tokenizer.decode(input_token_ids)
    logger.info(f"text: {text}")

if __name__ == "__main__":
    main()
