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

import pandas as pd
import torch
from datasets import load_dataset, Dataset
from unsloth import FastLanguageModel
from unsloth.chat_templates import standardize_sharegpt
from trl import SFTTrainer, SFTConfig
from transformers import TextStreamer

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]
os.environ['LOG_NAME'] = LOGGING_LABEL
from utils.log_util import logger


# ------------------------------
# Model
# ------------------------------
# load model
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

# ------------------------------
# Data prepare
# ------------------------------
# load dataset
reasoning_dataset = load_dataset("unsloth/OpenMathReasoning-mini", split="cot")
non_reasoning_dataset = load_dataset("mlabonne/FineTome-100k", split="train")
logger.info(f"reasoning_dataset: \n{reasoning_dataset}")
logger.info(f"non_reasoning_dataset: \n{non_reasoning_dataset}")

def generate_conversation(examples):
    """
    convert reasoning dataset into conversation format
    """
    problems = examples["problem"]
    solutions = examples["generate_conversation"]
    conversations = []
    for problem, solution in zip(problems, solutions):
        conversations.append([
            {"role": "user", "content": problem},
            {"role": "assistant", "content": solution}
        ])
    
    return {"conversations": conversations}

# convert reasoning dataset into conversation format
reasoning_conversations = tokenizer.apply_chat_template(
    reasoning_dataset.map(generate_conversation, batched=True)["conversations"],
    tokenize=False
)
logger.info(f"reasoning_conversation[0]: \n{reasoning_conversations[0]}")
logger.info(f"reasoning_conversations length: {len(reasoning_conversations)}")

# convert non reasoning dataset into conversation format
dataset = standardize_sharegpt(non_reasoning_dataset)  # fix up the format of the dataset
non_reasoning_conversations = tokenizer.apply_chat_template(
    dataset["conversations"],
    tokenize=False
)
logger.info(f"non_reasoning_conversation[0]: \n{non_reasoning_conversations[0]}")
logger.info(f"non_reasoning_conversations length: {len(non_reasoning_conversations)}")

# define some mixture of both sets of data(75% reasoning, 25% chat based)
chat_percentage = 0.25
non_reasoning_subset = pd.Series(non_reasoning_conversations)
non_reasoning_subset = non_reasoning_subset.sample(
    int(len(reasoning_conversations) * (chat_percentage / (1 - chat_percentage)))
)
logger.info(f"reasoning_conversations length: {len(reasoning_conversations)}")
logger.info(f"non_reasoning_conversations length: {len(non_reasoning_subset)}")

# combine both datasets
data = pd.concat([
    pd.Series(reasoning_conversations),
    pd.Series(non_reasoning_subset)
])
data.name = "text"
data = pd.DataFrame(data)
combined_dataset = Dataset.from_pandas(data)
combined_dataset = combined_dataset.shuffle(seed=3407)

# ------------------------------
# Train model
# ------------------------------
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = combined_dataset,
    eval_dataset = None,
    args = SFTConfig(
        dataset_text_field = "text",
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,  # use GA to minic batch size
        warmup_step = 5,
        # num_train_epochs = 1,
        max_steps = 30,
        learning_rate = 2e-4,  # reduce to 2e-5 for long training runs
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        report_to = "none",  # use this for WandB ect
    )
)

# show current memory stats
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024**3, 3)
max_memory = round(gpu_stats.total_memory / 1024**3, 3)
logger.info(f"GPU: {gpu_stats}")
logger.info(f"start_gpu_memory: {start_gpu_memory} GB")
logger.info(f"max_memory: {max_memory} GB")

# model training
trainer_stats = trainer.train()

# show final memory and time stats
used_memory = round(torch.cuda.max_memory_reserved() / 1024**3, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory / max_memory * 100, 3)
lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
logger.info(f"{round(trainer_stats.metrics['train_runtime'] / 60, 2)} minutes used for training.")
logger.info(f"Peak reserved memory = {used_memory} GB.")
logger.info(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
logger.info(f"Peak reserved memory % of max memory = {used_percentage} %.")
logger.info(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

# ------------------------------
# Inference
# ------------------------------
# non Thinking
messages = [
    {"role": "user", "content": "Solve (x + 2)^2 = 0."}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,
)
_ = model.generate(
    **tokenizer(text, return_tensors="pt").to("cuda"),
    max_new_tokens=256,
    temperature=0.7, top_p=0.80, top_k=20,  # for non thinking
    streamer = TextStreamer(tokenizer, skip_prompt=True), 
)

# Thinking
messages = [
    {"role" : "user", "content" : "Solve (x + 2)^2 = 0."}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize = False,
    add_generation_prompt = True, # Must add for generation
    enable_thinking = True, # Disable thinking
)

from transformers import TextStreamer
_ = model.generate(
    **tokenizer(text, return_tensors="pt").to("cuda"),
    max_new_tokens=1024,
    temperature = 0.6, top_p = 0.95, top_k = 20, # for thinking
    streamer = TextStreamer(tokenizer, skip_prompt=True),
)


# ------------------------------
# Saving and Loading finetuned models
# ------------------------------
# huggingface repo save
# -------------------------
model.push_to_hub("wangzf/Qwen3-14B-Reasoning", token="")
tokenizer.push_to_hub("wangzf/Qwen3-14B-Reasoning", token="")

# local save
# -------------------------
model.save_pretrained("lora_model")
tokenizer.save_pretrained("lora_model")

# load LoRA adapters
# -------------------------
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="lora_model",
    max_seq_length=2048,
    load_in_4bit=True,
)

# save to float16 for VLLM
# -------------------------
# Merge to 16bit
if True:
    model.save_pretrained_merged("model", tokenizer, save_method = "merged_16bit",)
if True: # Pushing to HF Hub
    model.push_to_hub_merged("hf/model", tokenizer, save_method = "merged_16bit", token = "")

# Merge to 4bit
if True:
    model.save_pretrained_merged("model", tokenizer, save_method = "merged_4bit",)
if True: # Pushing to HF Hub
    model.push_to_hub_merged("hf/model", tokenizer, save_method = "merged_4bit", token = "")

# Just LoRA adapters
if True:
    model.save_pretrained_merged("model", tokenizer, save_method = "lora",)
if True: # Pushing to HF Hub
    model.push_to_hub_merged("hf/model", tokenizer, save_method = "lora", token = "")

# GGUF/llama.cpp Conversion
# -------------------------
# Save to 8bit Q8_0
if True:
    model.save_pretrained_gguf("model", tokenizer,)
if True:
    model.push_to_hub_gguf("hf/model", tokenizer, token = "")

# Save to 16bit GGUF
if True:
    model.save_pretrained_gguf("model", tokenizer, quantization_method = "f16")
if True: # Pushing to HF Hub
    model.push_to_hub_gguf("hf/model", tokenizer, quantization_method = "f16", token = "")

# Save to q4_k_m GGUF
if True:
    model.save_pretrained_gguf("model", tokenizer, quantization_method = "q4_k_m")
if True: # Pushing to HF Hub
    model.push_to_hub_gguf("hf/model", tokenizer, quantization_method = "q4_k_m", token = "")

# Save to multiple GGUF options - much faster if you want multiple!
if True:
    model.push_to_hub_gguf(
        "hf/model", # Change hf to your username!
        tokenizer,
        quantization_method = ["q4_k_m", "q8_0", "q5_k_m",],
        token = "", # Get a token at https://huggingface.co/settings/tokens
    )




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
