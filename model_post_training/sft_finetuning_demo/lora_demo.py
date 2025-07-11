# -*- coding: utf-8 -*-

# ***************************************************
# * File        : test.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2024-09-27
# * Version     : 0.1.092717
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


from transformers import AutoModelForSeq2SeqLM
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]

# model path
model_name_or_path = "bigscience/mt0-large"
tokenizer_namme_or_path = "bigscience/mt0-large"

# model
model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)

# peft config
peft_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
)
model = get_peft_model(model, peft_config)




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
