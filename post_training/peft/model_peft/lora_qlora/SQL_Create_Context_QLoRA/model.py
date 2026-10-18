# -*- coding: utf-8 -*-

# ***************************************************
# * File        : model.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-07-19
# * Version     : 1.0.071918
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

import torch
from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    AutoTokenizer,
)
from trl import setup_chat_format, SFTTrainer

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]
os.environ['LOG_NAME'] = LOGGING_LABEL
from utils.log_util import logger


def get_model_tokenizer():
    # Hugging Face model id
    model_id = "codellama/CodeLlama-7b-hf"  # or 'mistralai/Mistral-7B-v1.0'

    # BitsAndBytesConfig int-4 config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        attn_implementation="flash_attention_2",
        torch_type=torch.bfloat16,
        quantization_config=bnb_config,
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.padding_side = "right"

    # set chat template to OAI chatML, remove if you start from a fine-tuned model
    model, tokenizer = setup_chat_format(model, tokenizer)

    return model, tokenizer




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
