# -*- coding: utf-8 -*-

# ***************************************************
# * File        : data_load.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2025-02-22
# * Version     : 0.1.022201
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
    
from utils.log_util import logger

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


def load_instruction_data(data_path: str):
    """
    load instruction entries json data
    """
    # instruction entries json data
    with open(data_path, "r") as file:
        data = json.load(file)

    return data




# 测试代码 main 函数
def main():
    inst_entries_with_pref_path = "./dataset/finetuning/instruction-preference-data.json"
    inst_entries_with_pref_data = load_instruction_data(
        data_path =inst_entries_with_pref_path 
    )
    logger.info(f"Number of entries: {len(inst_entries_with_pref_data)}")

if __name__ == "__main__":
    main()
