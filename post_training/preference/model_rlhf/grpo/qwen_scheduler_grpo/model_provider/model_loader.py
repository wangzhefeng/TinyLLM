# -*- coding: utf-8 -*-

# ***************************************************
# * File        : model_loader.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-05-05
# * Version     : 1.0.050502
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

__all__ = [
    "model_load"
]

# python libraries
import os
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)


from unsloth import FastLanguageModel

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


def model_load(max_seq_length: int = 2048):
    # max_seq_length = 2048  # can increase for longer reasoning traces
    lora_rank = 32  # larger rank = smarter, but slower

    # original model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="Qwen/Qwen2.5-Coder-7B-Instruct",
        max_seq_length=max_seq_length,
        load_in_4bit=True,  # False for LoRA 16bit
        fast_inference=True,  # Enable vLLM fast inference
        max_lora_rank=lora_rank,
        gpu_memory_utilization=0.85,  # Reduce if out of memory
    )

    # Lora fine-tuning
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,  # choose any number>0, suggested 8, 16, 32, 64, 128
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],  # remove QKVO if out of memory
        lora_alpha=lora_rank,
        use_gradient_checkpointing="unsloth",  # enable long context finetuning
        random_state=3407,
    )
    
    return model, tokenizer




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
