# -*- coding: utf-8 -*-

# ***************************************************
# * File        : data_config.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-03-16
# * Version     : 1.0.031601
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


instruction_data_path = "./dataset/finetune/instruction-data.json"
logger.info(f"instruction_data_path: {instruction_data_path}")

instruction_data_with_preference_path = "./dataset/finetune/instruction-preference-data.json"
logger.info(f"instruction_data_with_preference_path: {instruction_data_with_preference_path}")




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
