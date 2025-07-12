# -*- coding: utf-8 -*-

# ***************************************************
# * File        : load_llama3_pretrained_weights.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-04-04
# * Version     : 1.0.040417
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
from models.llama3_8B import Model
from model_load.model_cfgs import LLAMA3_CONFIG_8B
from model_load.meta_llama3_weights_load_hf import load_weights_into_llama

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


# model params
LLAMA3_CONFIG_8B = DotDict(LLAMA3_CONFIG_8B)


def model_with_llama3_weights(weights):
    # device
    device = device_setting()

    # model load
    model = Model(LLAMA3_CONFIG_8B)
    model = load_weights_into_llama(model, weights, LLAMA3_CONFIG_8B)
    model.to(device)

    return model, LLAMA3_CONFIG_8B




# 测试代码 main 函数
def main():
    from model_load.meta_llama3_weights_load_hf import (
        download_llama3_model, 
        download_llama3_instruct_model
    )
    from utils.llm.gpt_generate import generate
    from layers.tokenizers.tokenization import choose_tokenizer, text_to_token_ids, token_ids_to_text
    from layers.tokenizers.chat_format_tokenizer import ChatFormat
    from utils.device import device_setting, torch_gc
    from utils.log_util import logger

    # device
    device = device_setting()

    """
    # ------------------------------
    # Llama-3-8b model
    # ------------------------------
    # pretrained model weights
    weights = download_llama3_model(model_path = "downloaded_models/llama_model/Meta-Llama-3-8B")
    # build model
    model, LLAMA3_CONFIG_8B = model_with_llama3_weights(weights = weights)
    # model inference
    token_ids = generate(
        model=model,
        token_idx=text_to_token_ids("Every effort", tokenizer_model="llama3-8B").to(device),
        max_new_tokens=25,
        context_size=LLAMA3_CONFIG_8B.context_length,
        temperature=0.0,
        top_k=1,
        eos_id=50256,  # TODO
    )
    logger.info(f"Output text: \n{token_ids_to_text(token_ids, tokenizer_model='llama3-8B')}")
    
    # garbage collection
    del weights
    torch_gc()
    """
    # ------------------------------
    # Llama-3-8B instruct model
    # ------------------------------
    # pretrained model weights
    weights_chat = download_llama3_instruct_model(model_path = "downloaded_models/llama_model/Meta-Llama-3-8B-Instruct")
    # build model
    model, LLAMA3_CONFIG_8B = model_with_llama3_weights(weights = weights_chat)
    # chat prompt
    tokenizer = choose_tokenizer(tokenizer_model = "llama3-8B")
    chat_prompt = ChatFormat(tokenizer = tokenizer)
    # model inference
    token_ids = generate(
        model=model,
        token_idx=text_to_token_ids("What do llamas eat?", tokenizer_model=chat_prompt).to(device),
        max_new_tokens=25,
        context_size=LLAMA3_CONFIG_8B.context_length,
        temperature=0.0,
        top_k=1,
        eos_id=50256,  # TODO
    )
    output_text = token_ids_to_text(token_ids, tokenizer_model='llama3-8B')
    logger.info(f"Output text: \n{output_text}")

    def clean_text(text, header_end = "assistant<|end_header_id|>\n\n"):
        # Find the index of the first occurrence of "<|end_header_id|>"
        index = text.find(header_end)
        if index != -1:
            # Return the substring starting after "<|end_header_id|>"
            return text[index + len(header_end):].strip()  # Strip removes leading/trailing whitespace
        else:
            # If the token is not found, return the original text
            return text

    logger.info(f"Output text:\n{clean_text(output_text)}")

    # garbage collection
    del weights_chat
    torch_gc()

if __name__ == "__main__":
    main()
