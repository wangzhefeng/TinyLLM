# -*- coding: utf-8 -*-

# ***************************************************
# * File        : data_loader.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-01-23
# * Version     : 1.0.012321
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
import urllib.request

from utils.log_util import logger

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


def _data_download(url: str, file_path: str):
    """
    data download
    """
    # version 1
    urllib.request.urlretrieve(url, file_path)
    # version 2
    '''
    # download
    with urllib.request.urlopen(url) as response:
        text_data = response.read().decode("utf-8")
    # write
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(text_data)
    '''


def data_load(url: str = None, data_dir: str = "dataset/pretrain"):
    """
    data load
    """
    # 数据路径
    data_path = os.path.join(ROOT, data_dir)
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    # 数据文件路径
    file_path = os.path.join(data_path, url.split("/")[-1])
    # 数据下载、数据加载
    if not os.path.exists(file_path):
        # download
        logger.info(f"Download data...")
        _data_download(url, file_path)
        logger.info(f"Data has downloaded into '{data_path}'")
        # read
        logger.info(f"Load data...")
        with open(file_path, "r", encoding="utf-8") as file:
            raw_text = file.read()
        logger.info(f"Total number of character: {len(raw_text)}")
    else:
        # logger.info(f"Load data...")
        with open(file_path, "r", encoding="utf-8") as f:
            raw_text = f.read()
        # logger.info(f"Total number of character: {len(raw_text)}")

    return raw_text




# 测试代码 main 函数
def main():
    # llm pretrain data
    raw_text = data_load(
        url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"
    )
    logger.info(f"raw_text[:99]: {raw_text[:99]}")
    logger.info(f"raw_text[:99]: {raw_text[-99:]}")

if __name__ == "__main__":
    main()
