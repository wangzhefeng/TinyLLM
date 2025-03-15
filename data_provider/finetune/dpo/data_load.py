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

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


def load_dpo_data(data_path: str):
    """
    load instruction entries json data
    """
    # instruction entries json data
    with open(data_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    return data




# 测试代码 main 函数
def main():
    from data_provider.finetune.dpo.data_config import instruction_data_with_preference_path
    from utils.log_util import logger

    inst_entries_with_pref_data = load_dpo_data(data_path = instruction_data_with_preference_path)
    logger.info(f"Number of entries: {len(inst_entries_with_pref_data)}")

if __name__ == "__main__":
    main()
