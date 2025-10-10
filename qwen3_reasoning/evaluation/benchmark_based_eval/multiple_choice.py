# -*- coding: utf-8 -*-

# ***************************************************
# * File        : multiple_choice.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-10-05
# * Version     : 1.0.100516
# * Description : Evaluating answer-choice accuracy
# * Link        : https://github.com/rasbt/reasoning-from-scratch/tree/main/chF/02_mmlu
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
from datasets import (
    load_dataset,
    get_dataset_config_names,
)

from qwen3_reasoning.llm_basic.qwen3_06B import (
    device, 
    QWEN3_CONFIG,
    model, 
    tokenizer,
)
from layers.inference import generate_qwen3_stream

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]
os.environ['LOG_NAME'] = LOGGING_LABEL
from utils.log_util import logger


def load_data(verbose=False):
    configs = get_dataset_config_names("cais/mmlu")
    dataset = load_dataset("cais/mmlu", "high_school_mathematics")

    # Inspect the first example from the test set
    if verbose:
        example = dataset["test"][0]
        (example)
    
    return configs, dataset


def format_prompt(example):
    """
    Loading a pre-trained model
    """
    prompt = (
        f"{example['question']}\n"
        f"A. {example['choices'][0]}\n"
        f"B. {example['choices'][1]}\n"
        f"C. {example['choices'][2]}\n"
        f"D. {example['choices'][3]}\n"
        "Answer: "  # Trailing space in "Answer: " encourages a single-letter next token
    )

    return prompt


def get_prompt(prompt, tokenizer, device):
    prompt_ids = tokenizer.encode(prompt)
    prompt_fmt = torch.tensor(prompt_ids, device=device)
    # Add batch dimension
    prompt_fmt = prompt_fmt.unsqueeze(0)

    return prompt_fmt


def predict_choice(model, tokenizer, prompt_fmt, max_new_tokens=8):
    pred = None
    for t in generate_qwen3_stream(
        model=model,
        token_ids=prompt_fmt,
        max_new_tokens=max_new_tokens,
        context_length=QWEN3_CONFIG.context_length,
        temperature=0.0,
        top_k=None,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
    ):
        answer = tokenizer.decode(t.squeeze(0).tolist())
        for letter in answer:
            letter = letter.upper()
            # stop as soon as a letter appears
            if letter in "ABCD":
                pred = letter
                break
        if pred:
            break
    
    return pred




# 测试代码 main 函数
def main():
    # prompt test
    example = {
        "question": (
            "How many ways are there to put 4 distinguishable"
            " balls into 2 indistinguishable boxes?"
        ),
        "choices": ["7", "11", "16", "8"],
        "answer": "D",
    }
    prompt = format_prompt(example)
    print(f"prompt: \n{prompt}")
    
    prompt_fmt = get_prompt(prompt=prompt, tokenizer=tokenizer, device=device)
    print(f"prompt_fmt: \n{prompt_fmt}")
    
    # test
    pred = predict_choice(model=model, tokenizer=tokenizer, prompt_fmt=prompt_fmt, max_new_tokens=8)
    print(
        f"Generated letter: {pred}\n"
        f"Correct? {pred == example['answer']}"
    )

if __name__ == "__main__":
    main()
