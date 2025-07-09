sss# -*- coding: utf-8 -*-

# ***************************************************
# * File        : train_grpo.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-05-04
# * Version     : 1.0.050421
# * Description : description
# * Link        : https://huggingface.co/anakin87/qwen-scheduler-7b-grpo
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
# global variable
LOGGING_LABEL = __file__.split('\\')[-1][:-3]
os.environ['LOG_NAME'] = LOGGING_LABEL

import wandb
from trl import GRPOConfig, GRPOTrainer

from grpo.qwen_scheduler_grpo.data_provider.data_loader import data_load
from grpo.qwen_scheduler_grpo.model_provider.model_loader import model_load
from grpo.qwen_scheduler_grpo.model_provider.reward_func import (
    format_reward, 
    sorted_events_reward, 
    score_reward
)
from utils.log_util import logger


# wandb monitor
wandb.init(project="GRPO-reboost")

# Load the original model
model, tokenizer = model_load()

# Dataset preparation
ds = data_load()

# Traiing configuration
tokenized_prompts = [
    tokenizer.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True)
    for prompt in ds["prompt"]
]
exact_max_prompt_length = max(
    [len(tokenized_prompt) for tokenized_prompt in tokenized_prompts]
)
max_prompt_length = 448
max_seq_length = 2048
new_model_id = "wangzf/qwen-scheduler-7b-grpo"

training_args = GRPOConfig(
    learning_rate=8e-6,
    adam_beta1=0.9,
    adam_beta2=0.99,
    weight_decay=0.1,
    warmup_ratio=0.01,
    lr_scheduler_type="cosine",
    optim="paged_adamw_8bit",
    logging_steps=1,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=1,
    num_generations=8,  # Decrease if out of memory
    max_prompt_length=max_prompt_length,
    max_completion_length=max_seq_length - max_prompt_length,
    max_grad_norm=0.1,
    report_to="wandb",
    output_dir="outputs",
    overwrite_output_dir=True,
    push_to_hub=True,
    hub_model_id=new_model_id,
    hub_strategy="every_save",
    save_strategy="steps",
    save_steps=50,
    save_total_limit=1,
    num_train_epochs=3,
)

# Training
trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[
        format_reward,
        sorted_events_reward,
        score_reward,
    ],
    args=training_args,
    train_dataset=ds,
)
trainer.train()




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
