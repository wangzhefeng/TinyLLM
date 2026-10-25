# -*- coding: utf-8 -*-

# ***************************************************
# * File        : hf_finetuning.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-07-08
# * Version     : 1.0.070816
# * Description : description
# * Link        : https://www.philschmid.de/fine-tune-llms-in-2024-with-trl#1-define-our-use-case
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
from transformers import TrainingArguments
from trl import SFTTrainer
from peft import LoraConfig, AutoPeftModelForCausalLM

from data_factory import get_dataset
from model import get_model_tokenizer
from utils.online import hf_hub_login, wandb_login

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]
os.environ['LOG_NAME'] = LOGGING_LABEL
from utils.log_util import logger




# 测试代码 main 函数
def main():
    # env
    # hugging face login
    hf_hub_login()

    # wandb login
    wandb_login()

    # dataset
    train_dataset = get_dataset()

    # model and tokenizer
    model, tokenizer = get_model_tokenizer()

    # LoRA config based on QLoRA paper & Sebastian Raschka experiment
    peft_config = LoraConfig(
        r=256,
        lora_alpha=128,
        lora_dropout=0.05,
        bias="none",
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )

    # training config
    training_args = TrainingArguments(
        output_dir="code-llama-7b-text-to-sql", # directory to save and repository id
        num_train_epochs=3,                     # number of training epochs
        per_device_train_batch_size=3,          # batch size per device during training
        gradient_accumulation_steps=2,          # number of steps before performing a backward/update pass
        gradient_checkpointing=True,            # use gradient checkpointing to save memory
        optim="adamw_torch_fused",              # use fused adamw optimizer
        logging_steps=10,                       # log every 10 steps
        save_strategy="epoch",                  # save checkpoint every epoch
        learning_rate=2e-4,                     # learning rate, based on QLoRA paper
        bf16=True,                              # use bfloat16 precision
        tf32=True,                              # use tf32 precision
        max_grad_norm=0.3,                      # max gradient norm based on QLoRA paper
        warmup_ratio=0.03,                      # warmup ratio based on QLoRA paper
        lr_scheduler_type="constant",           # use constant learning rate scheduler
        push_to_hub=True,                       # push model to hub
        report_to="tensorboard",                # report metrics to tensorboard
    )

    # Trainer 
    max_seq_length = 3072  # max sequence length for model and packaging of the dataset
    trainer = SFTTrainer(
        model=model, 
        args=training_args,
        train_dataset=train_dataset,
        peft_config=peft_config,
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        packing=True,
        dataset_kwargs={
            "add_special_tokens": False,  # template with special tokens
            "append_concat_token": False,  # No need to add additional separator token
        }
    )

    # Model training
    trainer.train()
    
    # Model saving
    trainer.save_model()
    
    # Free memory
    del model
    del trainer
    torch.cuda.empty_cache()

    # Merge peft and base model
    # Load PEFT model on CPU
    model = AutoPeftModelForCausalLM.from_pretrained(
        training_args.output_dir,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    # Merge LoRA and base model and save
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(
        training_args.output_dir,safe_serialization=True, 
        max_shard_size="2GB"
    )

if __name__ == "__main__":
    main()
