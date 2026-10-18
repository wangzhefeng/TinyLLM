# -*- coding: utf-8 -*-

# ***************************************************
# * File        : main.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-07-17
# * Version     : 1.0.071700
# * Description : description
# * Link        : https://www.philschmid.de/fine-tune-llms-in-2025
# *               https://huggingface.co/datasets/microsoft/orca-math-word-problems-200k
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

from datasets import load_dataset

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]
os.environ['LOG_NAME'] = LOGGING_LABEL
from utils.log_util import logger


# ------------------------------
# data
# ------------------------------
# Create system prompt
system_message = """Solve the given high school math problem by providing a clear explanation of each step leading to the final solution.
 
Provide a detailed breakdown of your calculations, beginning with an explanation of the problem and describing how you derive each formula, value, or conclusion. Use logical steps that build upon one another, to arrive at the final answer in a systematic manner.
 
# Steps
 
1. **Understand the Problem**: Restate the given math problem and clearly identify the main question and any important given values.
2. **Set Up**: Identify the key formulas or concepts that could help solve the problem (e.g., algebraic manipulation, geometry formulas, trigonometric identities).
3. **Solve Step-by-Step**: Iteratively progress through each step of the math problem, justifying why each consecutive operation brings you closer to the solution.
4. **Double Check**: If applicable, double check the work for accuracy and sense, and mention potential alternative approaches if any.
5. **Final Answer**: Provide the numerical or algebraic solution clearly, accompanied by appropriate units if relevant.
 
# Notes
 
- Always clearly define any variable or term used.
- Wherever applicable, include unit conversions or context to explain why each formula or step has been chosen.
- Assume the level of mathematics is suitable for high school, and avoid overly advanced math techniques unless they are common at that level.
"""

def create_conversation(sample):
    """
    convert to messages
    """
    return {
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": sample["question"]},
            {"role": "assistant", "content": sample["answer"]},
        ]
    }


def get_dataset():
    # data path
    data_path = Path(ROOT).joinpath("./dataset/Orca-math-word-problems-SFT")
    data_path.mkdir(exist_ok=True, parents=True)
    train_path = data_path.joinpath("train_dataset.json")
    test_path = data_path.joinpath("test_dataset.json")
    if not train_path.exists() and not test_path.exists():
        # Load from huggingface hub
        dataset = load_dataset(
            "microsoft/orca-math-word-problems-200k", 
            split="train",
            cache_dir=data_path,
        )
        # Convert dataset to OAI messages
        dataset = dataset.map(
            create_conversation, 
            remove_columns=dataset.features, 
            batched=False
        )
        # save datasets to disk
        dataset.to_json(train_path, orient="records")
    # TODO Load dataset
    # train_dataset = load_dataset("json", data_files=str(train_path), split="train")

    # return train_dataset




# 测试代码 main 函数
def main():
    get_dataset()

if __name__ == "__main__":
    main()
