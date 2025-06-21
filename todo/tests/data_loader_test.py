# -*- coding: utf-8 -*-

# ***************************************************
# * File        : data_loader_test.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2025-02-10
# * Version     : 0.1.021023
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import os
import sys
ROOT = str(os.getcwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)
from pathlib import Path

from data_provider.pretrain.data_loader import create_dataloader
from utils.log_util import logger

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]




# 测试代码 main 函数
def main():
    import tiktoken
    from data_provider.pretrain.data_load import data_load

    # ------------------------------
    # data download & load
    # ------------------------------
    raw_text = data_load(
        url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"
    )

    # ------------------------------
    # tokenization
    # ------------------------------
    tokenizer = tiktoken.get_encoding("gpt2")
    enc_text = tokenizer.encode(raw_text) 
    logger.info(f"Pre-train token size: {len(enc_text)}")
    
    # ------------------------------
    # data sampling
    # ------------------------------  
    # token id
    enc_sample = enc_text[50:]

    # context size
    context_size = 4
 
    # data sampling
    x = enc_sample[:context_size]
    y = enc_sample[1:context_size]
    logger.info(f"x: {x}")
    logger.info(f"y:      {y}")
    
    # sliding window token ids
    for i in range(1, context_size+1):
        context = enc_sample[:i]
        desired = enc_sample[i]
        logger.info(f"{context} ----> {desired}")

    # sliding window tokens
    for i in range(1, context_size+1):
        context = enc_sample[:i]
        desired = enc_sample[i]
        logger.info(f"{tokenizer.decode(context)} ----> {tokenizer.decode([desired])}")
    
    # ------------------------------
    # dataset and dataloader test
    # ------------------------------
    # params
    batch_size = 8
    max_length = 4
    # data loader
    dataloader = create_dataloader(
        raw_text,
        batch_size=batch_size,
        max_length=max_length,
        stride=max_length,
        shuffle=False,
        drop_last=True,
    )
    # data loader test
    for batch in dataloader:
        # data batch
        x, y = batch
        logger.info(f"x: \n{x}")
        logger.info(f"y: \n{y}")
        break

if __name__ == "__main__":
    main()
