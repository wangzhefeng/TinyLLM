# -*- coding: utf-8 -*-

# ***************************************************
# * File        : deploy.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-07-17
# * Version     : 1.0.071700
# * Description : description
# * Link        : https://huggingface.co/blog/inference-endpoints-llm
# *               https://huggingface.co/blog/sagemaker-huggingface-llm
# *               https://github.com/huggingface/text-generation-inference
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

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]
os.environ['LOG_NAME'] = LOGGING_LABEL
from utils.log_util import logger

import requests as r
from transformers import AutoTokenizer
from datasets import load_dataset
from random import randint
 
 
# Load our test dataset and Tokenizer again
tokenizer = AutoTokenizer.from_pretrained("code-llama-7b-text-to-sql")
eval_dataset = load_dataset("json", data_files="test_dataset.json", split="train")
rand_idx = randint(0, len(eval_dataset))
 
# generate the same prompt as for the first local test
prompt = tokenizer.apply_chat_template(eval_dataset[rand_idx]["messages"][:2], tokenize=False, add_generation_prompt=True)
request= {"inputs":prompt,"parameters":{"temperature":0.2, "top_p": 0.95, "max_new_tokens": 256}}
 
# send request to inference server
resp = r.post("http://127.0.0.1:8080/generate", json=request)
 
output = resp.json()["generated_text"].strip()
time_per_token = resp.headers.get("x-time-per-token")
time_prompt_tokens = resp.headers.get("x-prompt-tokens")
 
# Print results
print(f"Query:\n{eval_dataset[rand_idx]['messages'][1]['content']}")
print(f"Original Answer:\n{eval_dataset[rand_idx]['messages'][2]['content']}")
print(f"Generated Answer:\n{output}")
print(f"Latency per token: {time_per_token}ms")
print(f"Latency prompt encoding: {time_prompt_tokens}ms")





# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
