# -*- coding: utf-8 -*-

# ***************************************************
# * File        : preference_finetuning.py
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

import torch

from finetuning.direct_preference_optimization.data_load import load_instruction_data
from finetuning.instruction_follow.data_process import format_input_alpaca
from utils.log_util import logger

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


def _build_data(data_path: str, train_ratio: float = 0.85, test_ratio: float = 0.10):
    # data load
    data = load_instruction_data(data_path = data_path)
    logger.info(f"Number of entries: {len(data_path)}")
    logger.info(f"data[50]: {data[50]}")
    logger.info(f"data[999]: {data[999]}")
    model_input = format_input_alpaca(data[50])
    logger.info(f"model_input: {model_input}")
    desired_response = f"### Response:\n{data[50]['chosen']}"
    possible_response = f"### Response:\n{data[50]['rejected']}"
    logger.info(f"desired_response: {desired_response}")
    logger.info(f"possible_response: {possible_response}")
    # data split
    train_portion = int(len(data) * train_ratio)  # 85% 用作训练集
    test_portion = int(len(data) * test_ratio)    # 10% 用作测试集
    val_portion = len(data) - train_portion - test_portion  # 剩下的 5% 用作验证集
    train_data = data[:train_portion]
    test_data = data[train_portion:train_portion + test_portion]
    valid_data = data[train_portion + test_portion:]
    logger.info(f"Train set length: {len(train_data)}")
    logger.info(f"Test set length: {len(test_data)}")
    logger.info(f"Validation set length: {len(valid_data)}")

    return train_data, valid_data, test_data




# 测试代码 main 函数
def main():
    # data
    data_path = "./dataset/finetuning/instruction-preference-data.json"
    data = load_instruction_data(data_path = data_path)
    logger.info(f"Number of entries: {len(data)}")
    logger.info(f"data[50]: \n{data[50]}")
    logger.info(f"data[999]: \n{data[999]}")

    model_input = format_input_alpaca(data[50])
    logger.info(f"model_input: \n{model_input}")

    desired_response = f"### Response:\n{data[50]['chosen']}"
    possible_response = f"### Response:\n{data[50]['rejected']}"
    logger.info(f"desired_response: \n{desired_response}")
    logger.info(f"possible_response: \n{possible_response}")

if __name__ == "__main__":
    main()
