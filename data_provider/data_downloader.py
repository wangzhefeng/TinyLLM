# -*- coding: utf-8 -*-

# ***************************************************
# * File        : data_downloader.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-08-03
# * Version     : 1.0.080319
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
import urllib.request
from typing import List
import warnings
warnings.filterwarnings("ignore")

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]
os.environ['LOG_NAME'] = LOGGING_LABEL
from utils.log_util import logger


def download_file_if_absent(url: str, filename: str, search_dirs: List):
    # check if the file already exists
    for directory in search_dirs:
        file_path = os.path.join(directory, filename)
        if os.path.exists(file_path):
            logger.info(f"{filename} already exists in {file_path}")
            return file_path

    target_path = os.path.join(search_dirs[0], filename)
    try:
        with urllib.request.urlopen(url) as response, open(target_path, "wb") as out_file:
            out_file.write(response.read())
        logger.info(f"Downloaded {filename} to {target_path}")
    except Exception as e:
        logger.info(f"Failed to download {filename}. Error: {e}")
    
    return target_path




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
