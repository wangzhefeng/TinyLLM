# -*- coding: utf-8 -*-

# ***************************************************
# * File        : data_factory.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-07-19
# * Version     : 1.0.071918
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

from datasets import load_dataset

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]
os.environ['LOG_NAME'] = LOGGING_LABEL
from utils.log_util import logger


# convert dataset to OAI messages
system_message = """You are an text to SQL query translator. Users will ask you questions in English and you will generate a SQL query based on the provided SCHEMA.
SCHEMA:
{schema}"""


def create_conversation(sample):
    return {
        "messages": [
            {"role": "system", "content": system_message.format(schema=sample["context"])},
            {"role": "user", "content": sample["question"]},
            {"role": "assistant", "content": sample["answer"]}
        ]
    }


def get_dataset():
    # data path
    data_path = Path(ROOT).joinpath("./dataset/SQL-Create-Context-SFT")
    data_path.mkdir(exist_ok=True, parents=True)
    train_path = data_path.joinpath("train_dataset.json")
    test_path = data_path.joinpath("test_dataset.json")
    if not train_path.exists() and not test_path.exists():
        # load dataset from the huggingface hub
        dataset = load_dataset(
            "b-mc2/sql-create-context", 
            split="train", 
            cache_dir=data_path,
        )
        dataset = dataset.shuffle().select(range(12500))
        # Convert dataset to OAI messages
        dataset = dataset.map(
            create_conversation, 
            remove_columns=dataset.features, 
            batched=False,
        )
        # Split dataset into 10,000 training samples and 2,500 test samples
        dataset = dataset.train_test_split(test_size=2500/12500) 
        # Save datasets to disk
        dataset["train"].to_json(train_path, orient="records")
        dataset["test"].to_json(test_path, orient="records")
        # print(dataset.keys())
        # print(dataset["train"][345]["messages"])

    # TODO load dataset
    train_dataset = load_dataset("json", data_files=str(train_path), split="train")
    # test_dataset = load_dataset("json", data_files=str(test_path), split="test")
    
    return train_dataset#, test_dataset






# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
