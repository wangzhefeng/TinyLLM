# -*- coding: utf-8 -*-

# ***************************************************
# * File        : eval.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-07-17
# * Version     : 1.0.071700
# * Description : description
# * Link        : https://www.philschmid.de/evaluate-llm
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
from tqdm import tqdm
from random import randint
import warnings
warnings.filterwarnings("ignore")

import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer, pipeline
from datasets import load_dataset
 
# global variable
LOGGING_LABEL = Path(__file__).name[:-3]
os.environ['LOG_NAME'] = LOGGING_LABEL
from utils.log_util import logger


# ------------------------------
# model
# ------------------------------
# model path
peft_model_id = "./code-llama-7b-text-to-sql"
# peft_model_id = args.output_dir
 
# Load Model with PEFT adapter
model = AutoPeftModelForCausalLM.from_pretrained(
  peft_model_id,
  device_map="auto",
  torch_dtype=torch.float16
)
tokenizer = AutoTokenizer.from_pretrained(peft_model_id)

# load into pipeline
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

# ------------------------------
# dataset
# ------------------------------
# Load our test dataset
eval_dataset = load_dataset("json", data_files="test_dataset.json", split="train")
rand_idx = randint(0, len(eval_dataset))
 
# Test on sample
prompt = pipe.tokenizer.apply_chat_template(eval_dataset[rand_idx]["messages"][:2], tokenize=False, add_generation_prompt=True)
outputs = pipe(prompt, max_new_tokens=256, do_sample=False, temperature=0.1, top_k=50, top_p=0.1, eos_token_id=pipe.tokenizer.eos_token_id, pad_token_id=pipe.tokenizer.pad_token_id)
 
print(f"Query:\n{eval_dataset[rand_idx]['messages'][1]['content']}")
print(f"Original Answer:\n{eval_dataset[rand_idx]['messages'][2]['content']}")
print(f"Generated Answer:\n{outputs[0]['generated_text'][len(prompt):].strip()}")

# ------------------------------
# 
# ------------------------------
def evaluate(sample):
    prompt = pipe.tokenizer.apply_chat_template(sample["messages"][:2], tokenize=False, add_generation_prompt=True)
    outputs = pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95, eos_token_id=pipe.tokenizer.eos_token_id, pad_token_id=pipe.tokenizer.pad_token_id)
    predicted_answer = outputs[0]['generated_text'][len(prompt):].strip()
    if predicted_answer == sample["messages"][2]["content"]:
        return 1
    else:
        return 0
 
success_rate = []
number_of_eval_samples = 1000
# iterate over eval dataset and predict
for s in tqdm(eval_dataset.shuffle().select(range(number_of_eval_samples))):
    success_rate.append(evaluate(s))
 
# compute accuracy
accuracy = sum(success_rate)/len(success_rate)
print(f"Accuracy: {accuracy*100:.2f}%")
 


# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
