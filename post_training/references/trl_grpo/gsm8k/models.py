# -*- coding: utf-8 -*-

# ***************************************************
# * File        : train.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-07-20
# * Version     : 1.0.072016
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
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from modelscope import snapshot_download

from utils.device import device_setting
deivce = device_setting()

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]
os.environ['LOG_NAME'] = LOGGING_LABEL
from utils.log_util import logger


# model download
model_id = "Qwen/Qwen2.5-3B-Instruct"
local_path = snapshot_download(model_id, cache_dir="./downloaded_models")

# tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    local_path, 
    trust_remote_code=True
)
tokenizer.pad_token = tokenizer.eos_token

# base model
base_model = AutoModelForCausalLM.from_pretrained(
    local_path,
    torch_dtype = torch.float16,
    device_map = "auto",
    # attn_implementation = "flash_attention_2",
    trust_remote_code = True,
)
base_model.gradient_checkpointing_enable()

# LoRA config
lora_cfg = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj", 'k_proj', "v_proj", "o_proj", 
        "gate_proj", "up_proj", "down_proj",
    ],
)

# LoRA model
model = get_peft_model(base_model, lora_cfg)
model.print_trainable_parameters()




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
