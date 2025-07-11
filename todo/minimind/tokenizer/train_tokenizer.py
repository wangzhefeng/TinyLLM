# -*- coding: utf-8 -*-

# ***************************************************
# * File        : train_tokenizer.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-02-09
# * Version     : 0.1.020916
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
import json
import random
from tqdm import tqdm


from datasets import load_dataset
from transformers import AutoTokenizer
from tokenizers import (
    Tokenizer,
    models,
    pre_tokenizers,
    trainers,
    decoders,
)

from utils.log_util import logger

random.seed(42)

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


def _read_texts_from_jsonl(file_path):
    """
    读取 JSONL 文件并提取文本数据
    """
    with open(file_path, "r", encoding = "utf-8") as f:
        for line in f:
            data = json.loads(line)
            yield data["text"]


def train_tokenizer(data_path: str = "./minimind/dataset/pretrain_hq.jsonl", 
                    tokenizer_dir: str = "./minimind/tokenizer/model/",
                    tokenizer_config_dir: str = "./minimind/tokenizer/config/"):
    """
    train tokenizer
    """
    # ------------------------------
    # read training data
    # ------------------------------
    texts = _read_texts_from_jsonl(data_path)
    # ------------------------------
    # training tokenizer
    # ------------------------------ 
    # 初始化 tokenizer
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space = False)
    # 定义特殊 token
    special_tokens = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
    # 设置训练器并添加特殊 token
    trainer = trainers.BpeTrainer(
        vocab_size = 6400,
        special_tokens = special_tokens,
        show_progress = True,
        initial_alphabet = pre_tokenizers.ByteLevel.alphabet(),
    )
    # 训练 tokenizer
    tokenizer.train_from_iterator(texts, trainer = trainer)
    # 设置解码器
    tokenizer.decoder = decoders.ByteLevel()
    # 检查特殊 token 的索引
    assert tokenizer.token_to_id("<|endoftext|>") == 0
    assert tokenizer.token_to_id("<|im_start|>") == 1
    assert tokenizer.token_to_id("<|im_end|>") == 2
    # ------------------------------
    # save tokenizer
    # ------------------------------
    os.makedirs(tokenizer_dir, exist_ok=True)
    tokenizer.save(Path(tokenizer_dir).joinpath("tokenizer.json"))
    tokenizer.model.save(tokenizer_dir)
    
    # 手动创建配置文件
    config = {
        "add_bos_token": False,
        "add_eos_token": False,
        "add_prefix_space": False,
        "added_tokens_decoder": {
            "0": {
                "content": "<|endoftext|>",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True
            },
            "1": {
                "content": "<|im_start|>",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True
            },
            "2": {
                "content": "<|im_end|>",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True
            }
        },
        "additional_special_tokens": [],
        "bos_token": "<|im_start|>",
        "clean_up_tokenization_spaces": False,
        "eos_token": "<|im_end|>",
        "legacy": True,
        "model_max_length": 32768,
        "pad_token": "<|endoftext|>",
        "sp_model_kwargs": {},
        "spaces_between_special_tokens": False,
        "tokenizer_class": "PreTrainedTokenizerFast",
        "unk_token": "<|endoftext|>",
        "chat_template": "{% if messages[0]['role'] == 'system' %}{% set system_message = messages[0]['content'] %}{{ '<|im_start|>system\\n' + system_message + '<|im_end|>\\n' }}{% else %}{{ '<|im_start|>system\\nYou are a helpful assistant<|im_end|>\\n' }}{% endif %}{% for message in messages %}{% set content = message['content'] %}{% if message['role'] == 'user' %}{{ '<|im_start|>user\\n' + content + '<|im_end|>\\n<|im_start|>assistant\\n' }}{% elif message['role'] == 'assistant' %}{{ content + '<|im_end|>' + '\\n' }}{% endif %}{% endfor %}"
    }
    # 保存配置文件
    with open(Path(tokenizer_config_dir).joinpath("tokenizer_config.json"), "w", encoding = "utf-8") as config_file:
        json.dump(config, config_file, ensure_ascii = False, indent = 4)
    logger.info("Tokenizer training completed and saved")


def eval_tokenizer(tokenizer_dir):
    # 加载与训练的 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    # Prompt template
    messages = [
        {"role": "system", "content": "你是一个优秀的聊天机器人，总是给我正确的回应！"},
        {"role": "user", "content": "你来自哪里？"},
        {"role": "assistant", "content": "我来自地球"},
    ]
    new_prompt = tokenizer.apply_chat_template(messages, tokenize = False)
    logger.info(f"new_prompt: {new_prompt}")

    # 获取实际词汇表长度（包括特殊符号）
    actual_vocab_size = len(tokenizer)
    logger.info(f"tokenizer 实际词表长度: {actual_vocab_size}")
    
    model_inputs = tokenizer(new_prompt)
    logger.info(f"encoder 长度: {len(model_inputs['input_ids'])}")

    input_ids = model_inputs["input_ids"]
    response = tokenizer.decode(input_ids, skip_special_tokens=False)
    logger.info(f"decoder 和原始文本是否一致: {response == new_prompt}")




# 测试代码 main 函数
def main():
    data_path = "./minimind/dataset/pretrain_hq.jsonl"
    tokenizer_dir = "./minimind/tokenizer/model/"
    tokenizer_config_dir = "./minimind/tokenizer/config"
    train_tokenizer(data_path, tokenizer_dir, tokenizer_config_dir)
    # eval_tokenizer()

if __name__ == "__main__":
    main()
