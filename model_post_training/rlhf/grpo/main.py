# -*- coding: utf-8 -*-

# ***************************************************
# * File        : main.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-03-09
# * Version     : 1.0.030917
# * Description : https://colab.research.google.com/github/huggingface/notebooks/blob/main/course/en/chapter13/grpo_finetune.ipynb#scrollTo=2Kd8acEE88C4
# *               https://huggingface.co/learn/nlp-course/en/chapter12/4?fw=pt
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

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["SM_FRAMEWORK"] = "tf.keras"

import torch
import wandb
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import pipeline
from trl import GRPOConfig, GRPOTrainer

from utils.log_util import logger

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


# Log to weights & biases
wandb.login()

# dataset
dataset = load_dataset("mlabonne/smoltldr", cache_dir="./dataset/grpo/")
logger.info(f"dataset: \n{dataset}")

# model
model_id = "HuggingFaceTB/SmolLM-135M-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype = "auto",
    device_map = "auto",
    attn_implementation = "flash_attention_2",
    cache_dir="./downloaded_models/"
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# LoRA
lora_config =LoraConfig(
    task_type = "CAUSAL_LM",
    r = 16,
    lora_alpha = 32,
    target_modules = "all-linear",
)
model = get_peft_model(model, lora_config)
logger.info(f"model trainable parameters: {model.print_trainable_parameters()}")


'''
# reward function
def reward_len(completions, **kwargs):
    """
    Reward function
    """
    return [-abs(50 - len(completion)) for completion in completions]

# training arguments
training_args = GRPOConfig(
    output_dir = "GRPO",
    learning_rate = 2e-5,
    per_device_train_batch_size = 8,
    gradient_accumulation_steps = 2,
    max_prompt_length = 512,
    max_completion_length = 512,
    num_generations = 8,
    optim = "adamw_8bit",
    num_train_epochs = 1,
    bf16 = True,
    report_to = ["wandb"],
    remove_unused_columns = False,
    logging_steps = 1,
)

# Trainer
trainer = GRPOTrainer(
    model = model,
    reward_funcs = [reward_len],
    args = training_args,
    train_dataset = dataset["train"],
)

# train model
wandb.init(project = "GRPO")
trainer.train()

# push model to hub
merged_model = trainer.model.merge_and_unload()
merged_model.push_to_hub("", private = False)

# generate text
prompt = """
# A long document about the Cat

The cat (Felis catus), also referred to as the domestic cat or house cat, is a small 
domesticated carnivorous mammal. It is the only domesticated species of the family Felidae.
Advances in archaeology and genetics have shown that the domestication of the cat occurred
in the Near East around 7500 BC. It is commonly kept as a pet and farm cat, but also ranges
freely as a feral cat avoiding human contact. It is valued by humans for companionship and
its ability to kill vermin. Its retractable claws are adapted to killing small prey species
such as mice and rats. It has a strong, flexible body, quick reflexes, and sharp teeth,
and its night vision and sense of smell are well developed. It is a social species,
but a solitary hunter and a crepuscular predator. Cat communication includes
vocalizations—including meowing, purring, trilling, hissing, growling, and grunting—as
well as body language. It can hear sounds too faint or too high in frequency for human ears,
such as those made by small mammals. It secretes and perceives pheromones.
"""

messages = [
    {
        "role": "user", 
        "content": prompt
    },
]

# generator = pipeline("text-generation", model = "")
generator = pipeline("text-generation", model = model, tokenizer = tokenizer)
generate_kwargs = {
    "max_new_tokens": 256,
    "do_sample": True,
    "temperature": 0.5,
    "min_p": 0.1,
}
generated_text = generator(messages, generate_kwargs = generate_kwargs)
logger.info(f"generated text: {generated_text}")
'''



# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
