# -*- coding: utf-8 -*-

# ***************************************************
# * File        : load_llama2_pretrained_weights.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-03-30
# * Version     : 1.0.033020
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


import torch

from utils.args_tools import DotDict
from utils.device import device_setting
from models.llama2 import Model
from model_load.model_cfgs import LLAMA2_CONFIG_7B
from model_load.meta_llama2_weights_load_hf import load_weights_into_llama

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


# model params 
LLAMA2_CONFIG_7B = DotDict(LLAMA2_CONFIG_7B)


def model_with_llama2_weights(weights):
    # device
    device = device_setting()

    # model load
    model = Model(LLAMA2_CONFIG_7B)
    model = load_weights_into_llama(model, weights, LLAMA2_CONFIG_7B)
    model.to(device)

    return model, LLAMA2_CONFIG_7B




# 测试代码 main 函数
def main():
    from model_load.meta_llama2_weights_load_hf import (
        download_llama2_model, 
        download_llama2_chat_model
    )
    from utils.llm.gpt_generate import generate
    from layers.tokenizers.tokenization import text_to_token_ids, token_ids_to_text
    from utils.device import device_setting
    from utils.log_util import logger

    # device
    device = device_setting()

    # ------------------------------
    # Llama-2-7b model
    # ------------------------------
    weights = download_llama2_model(model_path = "downloaded_models/llama_model/Llama-2-7b")
    model, LLAMA2_CONFIG_7B = model_with_llama2_weights(weights = weights)
    # model inference
    token_ids = generate(
        model=model,
        token_idx=text_to_token_ids("Every effort moves", tokenizer_model="llama2").to(device),
        max_new_tokens=30,
        context_size=LLAMA2_CONFIG_7B.context_length,
        temperature=1.0,
        top_k=1,
        eos_id=50256,  # TODO
    )
    logger.info(f"Output text: \n{token_ids_to_text(token_ids, tokenizer_model='llama2')}")
    # ------------------------------
    # Llama-2-7b chat model
    # ------------------------------
    weights_chat = download_llama2_chat_model(model_path = "downloaded_models/llama_model/Llama-2-7b-chat")
    model, LLAMA2_CONFIG_7B = model_with_llama2_weights(weights = weights_chat)
    # model inference
    token_ids = generate(
        model=model,
        token_idx=text_to_token_ids("What do llamas eat?", tokenizer_model="llama2").to(device),
        max_new_tokens=30,
        context_size=LLAMA2_CONFIG_7B.context_length,
        temperature=1.0,
        top_k=1,
        eos_id=50256,  # TODO
    )
    logger.info(f"Output text: \n{token_ids_to_text(token_ids, tokenizer_model='llama2')}")

if __name__ == "__main__":
    main()
