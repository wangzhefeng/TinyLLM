# -*- coding: utf-8 -*-

# ***************************************************
# * File        : data_load.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-02-16
# * Version     : 0.1.021617
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import os
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)

import urllib.request

from utils.log_util import logger

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


def download_data(data_url, data_path):
    """
    数据下载
    """
    # data download
    if not Path(data_path).exists():
        with urllib.request.urlopen(data_url) as response:
            text_data = response.read().decode("utf-8")
        with open(data_path, "w", encoding = "utf-8") as file:
            file.write(text_data) 




# 测试代码 main 函数
def main(): 
    from data_provider.load_save_data import load_json_data
    from data_provider.finetune.instruction_follow.data_config import data_url, data_path

    # data download 
    download_data(data_url = data_url, data_path = data_path)
    
    # data load
    data = load_json_data(data_path = data_path)
    logger.info(f"Number of entries of instruction data: {len(data)}")
    logger.info(f"Example entry: \n{data[50]}")
    logger.info(f"Example entry: \n{data[999]}")

if __name__ == "__main__":
    main()
