# -*- coding: utf-8 -*-

# ***************************************************
# * File        : qwen_tokenizer.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-09-29
# * Version     : 1.0.092922
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

__all__ = []

# python libraries
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)
import warnings
warnings.filterwarnings("ignore")

from config.qwen3.model_download import download_from_huggingface
from layers.tokenizers.qwen3.qwen3_tokenizer import Qwen3Tokenizer

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


# tokenizer model path
model_dir = "./downloaded_models/qwen3_model/Qwen3-0.6B-base-for-reasoning"

# tokenizer model download
model_file_path = download_from_huggingface(
    kind="base", 
    tokenizer_only=True, 
    repo_id="rasbt/qwen3-from-scratch",
    revision="main",
    local_dir=model_dir,
)

# tokenizer
tokenizer = Qwen3Tokenizer(tokenizer_file_path=model_file_path)




# 测试代码 main 函数
def main():
    from utils.log_util import logger

    prompt = "Explain large language models."

    input_token_ids_list = tokenizer.encode(prompt)
    logger.info(f"input_token_ids_list: {input_token_ids_list}") 

    text = tokenizer.decode(input_token_ids_list)
    logger.info(f"text: {text}")

    eos_token_id = tokenizer.eos_token_id
    logger.info(f"token_id of <|endoftext|>: {tokenizer.encode('<|endoftext|>')}")
    logger.info(f"eos_token_id: {eos_token_id}")

    for i in input_token_ids_list:
        logger.info(f"{i} -> {tokenizer.decode([i])}")

if __name__ == "__main__":
    main()
