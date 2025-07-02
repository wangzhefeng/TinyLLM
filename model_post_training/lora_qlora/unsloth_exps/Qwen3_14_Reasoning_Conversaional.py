# -*- coding: utf-8 -*-

# ***************************************************
# * File        : Qwen3_14_Reasoning_Conversaional.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-07-02
# * Version     : 1.0.070223
# * Description : description
# * Link        : https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen3_(14B)-Reasoning-Conversational.ipynb#scrollTo=DjgH3lt0e2Sz
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
from unsloth import FastLanguageModel
from datasets import load_dataset

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]
os.environ['LOG_NAME'] = LOGGING_LABEL
from utils.log_util import logger


# model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen3-14B",
    max_seq_length=2048,
    load_in_4bit=True,
    load_in_8bit=False,
    full_finetuning=False,
    # token=None
)

# add LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r=32,
    target_modules=[
        "q_proj", "k_proj", "v_proj", 
        "o_proj", "gate_proj", "up_proj", "down_proj"
    ],
    lora_alpha=32,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=False,  # rank stabilized LoRA
    loftq_config=None,  # LoftQ
)

# data prep
reasoning_dataset = load_dataset("unsloth/OpenMathReasoning-mini", split="cot")
non_reasoning_dataset = load_dataset("mlabonne/FineTome-100k", split="train")



# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
