# -*- coding: utf-8 -*-

# ***************************************************
# * File        : load_save_data.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-03-16
# * Version     : 1.0.031615
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

__all__ = []

# python libraries
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)
import json
import urllib.request
from typing import Dict

import pandas as pd

from utils.log_util import logger

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


def load_json_data(data_path: str):
    """
    load json data
    """
    with open(data_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    print("Number of entries:", len(data))

    return data


def save_json_data(json_data: Dict, save_path: str):
    """
    save instruction entries with preference json data
    """
    with open(save_path, "w", encoding="utf-8") as file:
        json.dump(json_data, file, indent=4)


def load_csv_data(data_path: str):
    """
    load spam tsv data for finetuning text classification
    """
    df = pd.read_csv(data_path, sep="\t", header=None, names=["Label", "Text"])
    
    return df


def __data_download(url: str, file_path: str):
    """
    data download

    data url example: ("https://raw.githubusercontent.com/rasbt/"
               "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
               "the-verdict.txt")
    """
    # version 1
    urllib.request.urlretrieve(url, file_path)
    
    '''
    # version 2
    # download
    with urllib.request.urlopen(url) as response:
        text_data = response.read().decode("utf-8")
    # write
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(text_data)
    '''


def load_local_data(url: str=None, data_path: str=None, data_file: str=None):
    """
    data load
    """
    # 数据文件路径
    if url is not None:
        file_path = Path(data_path).joinpath(url.split("/")[-1])
    else:
        file_path = Path(data_path).joinpath(data_file)
    # 数据下载、数据加载
    if not Path(file_path).exists():
        # data download
        logger.info(f"Download train data...")
        __data_download(url, file_path)
        logger.info(f"Data has downloaded into '{data_path}'")
        # data read
        logger.info(f"Load train data...")
        with open(file_path, "r", encoding="utf-8") as file:
            raw_text = file.read()
        logger.info(f"Total number of character: {len(raw_text)}")
    else:
        logger.info(f"Load train data...")
        with open(file_path, "r", encoding="utf-8") as file:
            raw_text = file.read()

    return raw_text


def load_hf_data(data_path: str, data_name: str, cache_dir: str):
    """
    load huggingface dataset
    """
    from datasets import load_dataset
    dataset = load_dataset(data_path, data_name, cache_dir=cache_dir, split="train")
    # data combine
    all_text = ""
    all_data = dataset["page"]
    for example in all_data:
        all_text += "<page> "+ example + " </page>"

    return all_text




# 测试代码 main 函数
def main():
    dataset = load_hf_data(
        data_path="EleutherAI/wikitext_document_level", 
        data_name="wikitext-2-raw-v1",
        cache_dir="./dataset/pretrain",
    )
    print(type(dataset))

if __name__ == "__main__":
    main()
