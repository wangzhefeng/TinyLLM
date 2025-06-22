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
import os
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)

import json
from typing import Dict

import pandas as pd

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




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
