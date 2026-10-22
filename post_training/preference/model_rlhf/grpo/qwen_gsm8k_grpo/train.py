# -*- coding: utf-8 -*-

# ***************************************************
# * File        : train.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-07-20
# * Version     : 1.0.072017
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

from trl import GRPOConfig, GRPOTrainer

from .data_provider import get_dataset
from post_training.preference.model_rlhf.grpo.qwen_gsm8k_grpo.models import output_dir, tokenizer, model
from grpo import (
    soft_format_reward_func, 
    strict_format_reward_func,
    correctness_reward_func,
)

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]
os.environ['LOG_NAME'] = LOGGING_LABEL
from utils.log_util import logger


# data
dataset = get_dataset()

# finetuned model saved path
output_dir = Path("results/models/qwen3b-grpo-lora-fp16")
output_dir.mkdir(parents=True, exist_ok=True)

# training args
train_args = GRPOConfig(
    fp16=True,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    num_train_epochs=1,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    max_grad_norm=0.3,
    logging_steps=1,
    save_steps=100,
    output_dir=output_dir,
    report_to="tensorboard",
    max_prompt_length=512,
    max_completion_length=64,
    num_generations=8,
    use_vllm=False,
)

# trainer
trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[
        soft_format_reward_func,
        strict_format_reward_func,
        correctness_reward_func,
    ],
    args=train_args,
    train_dataset=dataset,
)

# model training
trainer.train()

# model save
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
logger.info(f"LoRA Adapter + Tokenizer 已保存到 {output_dir}.")




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
