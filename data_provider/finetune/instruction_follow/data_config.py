# -*- coding: utf-8 -*-

# ***************************************************
# * File        : data_config.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-03-09
# * Version     : 1.0.030914
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


from utils.log_util import logger

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


# instruction data url
data_url = (
    "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch"
    "/main/ch07/01_main-chapter-code/instruction-data.json"
)
logger.info(f"data_url: {data_url}")

# data dir
data_dir = "dataset/finetune/"
os.makedirs(data_dir, exist_ok=True)
logger.info(f"data_dir: {data_dir}")

# instruction data path
data_path = Path(data_dir).joinpath(data_url.split("/")[-1])
logger.info(f"data_path: {data_path}")




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
