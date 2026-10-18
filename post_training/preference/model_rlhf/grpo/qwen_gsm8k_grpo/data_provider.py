# -*- coding: utf-8 -*-

# ***************************************************
# * File        : data_provider_math.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-07-20
# * Version     : 1.0.072016
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
import re
import warnings
warnings.filterwarnings("ignore")

from datasets import load_dataset

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]
os.environ['LOG_NAME'] = LOGGING_LABEL
from utils.log_util import logger


system_prompt = "你是一个擅长用 XML 格式输出链式思考和答案的数学助理，使用这种格式进行回答<reasoning>...</reasoning>\n<answer>...</answer>"


def extract_final_answer(text: str) -> str:
    """
    加载 HuggingFace 上的 gsm8k 数学问答数据集，用于训练微调。
    使用正则表达式提取答案中 #### 后面的数字，作为最终答案。
    加载 HuggingFace 上的 gsm8k 数学问答数据集，用于训练微调。
    """
    match = re.search(r"####\s*(\d+)", text)
    if match:
        return match.group(1).strip()
    
    match = re.search(r"\\boxed\{(.*?)\}", text)
    if match:
        return match.group(1).strip()

    return ""


def preprocess(example):
    return {
        "prompt": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": example["question"]},
        ],
        "answer": extract_final_answer(example["answer"]),
    }


def get_dataset():
    # data path
    data_path = Path(ROOT).joinpath("./dataset/finetuning/math/gsm8k/gsm8k-math-SFT")
    data_path.mkdir(exist_ok=True, parents=True)
    train_path = data_path.joinpath("train_dataset.json")
    test_path = data_path.joinpath("test_dataset.json")
    if not train_path.exists() and not test_path.exists():
        # Load from huggingface hub
        dataset = load_dataset(
            "openai/gsm8k", 
            "main",
            split="train",
            cache_dir=data_path,
        )
        # Convert dataset to OAI messages
        dataset = dataset.map(
            preprocess,
            remove_columns=dataset.column_names, 
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
