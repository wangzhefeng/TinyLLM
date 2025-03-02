# -*- coding: utf-8 -*-

# ***************************************************
# * File        : data_load.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2025-02-16
# * Version     : 0.1.021617
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import os
import sys
ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
import json
import urllib.request

from utils.log_util import logger

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


def download_file(file_path, url):
    """
    数据下载
    """
    if not os.path.exists(file_path):
        with urllib.request.urlopen(url) as response:
            text_data = response.read().decode("utf-8")
        with open(file_path, "w", encoding = "utf-8") as file:
            file.write(text_data)
    # with open(file_path, "r", encoding = "utf-8") as file:
        # data = json.load(file)
    
    # return data


def load_file(file_path):
    """
    数据加载
    """
    with open(file_path, "r", encoding = "utf-8") as file:
        data = json.load(file)
    
    return data




# 测试代码 main 函数
def main():
    data_path = "./dataset/finetuning/"
    os.makedirs(data_path, exist_ok=True)
    file_path = os.path.join(data_path, "instruction-data.json")
    url = (
        "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch"
        "/main/ch07/01_main-chapter-code/instruction-data.json"
    )
    download_file(file_path, url)
    data = load_file(file_path)
    logger.info(f"Number of entries: {len(data)}")
    logger.info(f"Example entry: \n{data[50]}")
    logger.info(f"Example entry: \n{data[999]}")

if __name__ == "__main__":
    main()
