# -*- coding: utf-8 -*-

# ***************************************************
# * File        : data_load.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-01-01
# * Version     : 1.0.010120
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# * TODO        : 1.
# ***************************************************

__all__ = []

# python libraries
import os
import sys
ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from datasets import concatenate_datasets, load_dataset

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


def data_load():
    """
    data download or load
    """
    bookcorpus = load_dataset(
        path="bookcorpus", 
        split="train", 
        trust_remote_code=True
    )
    wiki = load_dataset(
        path="wikipedia", 
        name="20220301.en", 
        split="train", 
        trust_remote_code=True
    )
    # 仅保留 "text" 列
    wiki = wiki.remove_columns([col for col in wiki.column_names if col != "text"])
    # data concate
    dataset = concatenate_datasets([bookcorpus, wiki])
    
    return dataset


def data_split(dataset):
    """
    将数据集合切分为 90% 用于训练，10% 用于测试

    Args:
        dataset (_type_): _description_
    """
    dataset = dataset.train_test_split(test_size = 0.1)
    
    return dataset


def data_to_text(dataset, output_filename="data.txt"):
    """
    将数据集文本保存到磁盘的通用函数

    Args:
        dataset (_type_): _description_
        output_filename (str, optional): _description_. Defaults to "data.txt".
    """
    with open(output_filename, "w") as f:
        for t in dataset["text"]:
            print(t, file=f)







# 测试代码 main 函数
def main():
    # params
    data_dir = "/Users/wangzf/llm_proj/dataset/bert_data/"
    dataset = data_load()
    dataset = data_split(dataset)
    # 将训练集保存为 train.txt
    data_to_text(dataset["train"], os.path.join(data_dir, "train.txt"))
    # 将测试机保存为 test.txt
    data_to_text(dataset["test"], os.path.join(data_dir, "test.txt"))

if __name__ == "__main__":
    main()
