# -*- coding: utf-8 -*-

# ***************************************************
# * File        : tokenization_test.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-02-10
# * Version     : 0.1.021021
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import os
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)


from tokenizers.simple_custom import SimpleTokenizer
from tokenizers.simple_bpe import BPETokenizerSimple
from data_provider.pretrain.data_load import data_load
from utils.log_util import logger

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]




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

    # method 1: simple tokenizer
    # ---------------------------------------
    # 训练数据下载、加载
    raw_text = data_load(
        url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"
    )
    tokenizer = SimpleTokenizer(raw_text=raw_text)
    logger.info(f"tokenizer.n_vocab: {tokenizer.n_vocab}")
    
    token_ids = tokenizer.encode(text=input_text_2)
    logger.info(f"token_ids: {token_ids}")
    
    decoded_text = tokenizer.decode(tokens=token_ids)
    logger.info(f"decoded_text: {decoded_text}")
    
    # method 2: BPE: tiktoken
    # ---------------------------------------
    import tiktoken
    tokenizer = tiktoken.get_encoding("gpt2")
    logger.info(f"tokenizer.n_vocab: {tokenizer.n_vocab}")
    
    token_ids = tokenizer.encode(text=input_text_1, allowed_special={"<|endoftext|>"})
    logger.info(f"token_ids: {token_ids}")
    
    decoded_text = tokenizer.decode(tokens=token_ids)
    logger.info(f"decoded_text: {decoded_text}")

    # method 3: BPE: original implementation used in GPT-2
    # ---------------------------------------
    from tests.bpe_openai_gpt2 import get_encoder, download_vocab
    models_dir = "download_models"
    # donwload model
    download_vocab(models_dir = models_dir)
    # tokenizer
    tokenizer = get_encoder(model_name="gpt2_model", models_dir=models_dir)
    # logger.info(f"tokenizer.n_vocab: {tokenizer.n_vocab}")
    token_ids = tokenizer.encode(text=input_text_2)
    logger.info(f"token_ids: {token_ids}")
    decoded_text = tokenizer.decode(tokens=token_ids)
    logger.info(f"decoded_text: {decoded_text}")
    
    # method 4: BPE: huggingface transformers
    # ---------------------------------------
    from transformers import GPT2Tokenizer, GPT2TokenizerFast
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    # logger.info(f"tokenizer.n_vocab: {tokenizer.n_vocab}")
    token_ids = tokenizer(input_text_2)["input_ids"]
    logger.info(f"token_ids: {token_ids}")

    tokenizer_fast = GPT2TokenizerFast.from_pretrained("gpt2")
    # logger.info(f"tokenizer.n_vocab: {tokenizer.n_vocab}")
    token_ids_fast = tokenizer_fast(input_text_2)["input_ids"]
    logger.info(f"token_ids_fast: {token_ids_fast}")

if __name__ == "__main__":
    main()
