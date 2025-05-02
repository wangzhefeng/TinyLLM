# -*- coding: utf-8 -*-

# ***************************************************
# * File        : train_tokenizer.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2025-02-09
# * Version     : 0.1.020916
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import os
import sys
ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
import json
import random
from tqdm import tqdm

from datasets import load_dataset
from transformers import AutoTokenizer
from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer,
)

from utils.log_util import logger

random.seed(42)

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


def _read_texts_from_jsonl(file_path):
    """
    读取 JSONL 文件并提取文本数据
    """
    with open(file_path, "r", encoding = "utf-8") as f:
        for line in f:
            data = json.loads(line)
            yield data["text"]


def train_tokenizer(data_path = "./dataset/tokenizer/tokenizer_train.jsonl", 
                    tokenizer_dir: str = "./dataset/tokenizer/"):
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
    special_tokens = ["<unk>", "<s>", "</s>"]
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
    assert tokenizer.token_to_id("<unk>") == 0
    assert tokenizer.token_to_id("<s>") == 1
    assert tokenizer.token_to_id("</s>") == 2
    # ------------------------------
    # save tokenizer
    # ------------------------------
    os.makedirs(tokenizer_dir, exist_ok=True)
    tokenizer.save(os.path.join(tokenizer_dir, "tokenizer.json"))
    tokenizer.model.save(tokenizer_dir)
    
    # 手动创建配置文件
    config = {
        "add_bos_token": False,
        "add_eos_token": False,
        "add_prefix_space": True,
        "added_tokens_decoder": {
            "0": {
                "content": "<unk>",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True
            },
            "1": {
                "content": "<s>",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True
            },
            "2": {
                "content": "</s>",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True
            }
        },
        "additional_special_tokens": [],
        "bos_token": "<s>",
        "clean_up_tokenization_spaces": False,
        "eos_token": "</s>",
        "legacy": True,
        "model_max_length": 1000000000000000019884624838656,
        "pad_token": None,
        "sp_model_kwargs": {},
        "spaces_between_special_tokens": False,
        "tokenizer_class": "PreTrainedTokenizerFast",
        "unk_token": "<unk>",
        "use_default_system_prompt": False,
        "chat_template": "{% if messages[0]['role'] == 'system' %}{% set system_message = messages[0]['content'] %}{% endif %}{% if system_message is defined %}{{ system_message }}{% endif %}{% for message in messages %}{% set content = message['content'] %}{% if message['role'] == 'user' %}{{ '<s>user\\n' + content + '</s>\\n<s>assistant\\n' }}{% elif message['role'] == 'assistant' %}{{ content + '</s>' + '\\n' }}{% endif %}{% endfor %}"
    }
    # 保存配置文件
    with open(os.path.join(tokenizer_dir, "tokenizer_config.json"), "w", encoding = "utf-8") as config_file:
        json.dump(config, config_file, ensure_ascii = False, indent = 2)
    logger.info("Tokenizer training completed and saved.")


def eval_tokenizer():
    # 加载与训练的 tokenizer
    tokenizer = AutoTokenizer.from_pretrained("./model/minimind_tokenizer")
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
    response = tokenizer.decode(input_ids)
    logger.info(f"decoder 和原始文本是否一致: {response == new_prompt}")




# 测试代码 main 函数
def main():
    train_tokenizer()
    # eval_tokenizer()

if __name__ == "__main__":
    main()
