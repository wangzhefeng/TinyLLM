# -*- coding: utf-8 -*-

# ***************************************************
# * File        : models.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-07-20
# * Version     : 1.0.072018
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

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

from utils.device import device_setting
device = device_setting()

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]
os.environ['LOG_NAME'] = LOGGING_LABEL
from utils.log_util import logger


# model path
model_id = "HuggingFaceTB/SmolLM-135M-Instruct"

# tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    model_id, 
    trust_remote_code=True
)

# base model
base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",
    device_map="auto",
    # attn_implementation = "flash_attention_2",
    trust_remote_code=True,
    cache_dir = "./downloaded_models/"
)
base_model.gradient_checkpointing_enable()

# LoRA
lora_cfg = LoraConfig(
    r = 16,
    lora_alpha = 32,
    lora_dropout=0.05,
    bias="none",
    task_type = "CAUSAL_LM", 
    target_modules = "all-linear",
)

# LoRA model
model = get_peft_model(base_model, lora_cfg)
logger.info(f"model trainable parameters: {model.print_trainable_parameters()}")




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
