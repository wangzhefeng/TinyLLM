# -*- coding: utf-8 -*-

# ***************************************************
# * File        : data_process.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2025-02-16
# * Version     : 0.1.021619
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

from utils.log_util import logger

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


def format_input(entry):
    """
    format the input to the LLM use Alpaca-style prompt formatting

    Args:
        entry (_type_): _description_

    Returns:
        _type_: _description_
    """
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )
    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""
    # response_text = f"\n\n### Response:\n{entry['output']}"

    return (
        instruction_text + input_text 
        # + response_text
    )






# 测试代码 main 函数
def main():
    # data
    from finetuning.instruction_follow.data_load import load_file
    data = load_file(file_path = "./dataset/finetuning/instruction-data.json")
    
    # prompt format
    formated_entry = format_input(data[50])
    logger.info(f"format instruction entry: \n{formated_entry}")

    # prompt format
    formated_entry = format_input(data[999])
    logger.info(f"format instruction entry: \n{formated_entry}")

    # data split
    train_portion = int(len(data) * 0.85)
    test_portion = int(len(data) * 0.10)
    valid_portion = len(data) - train_portion - test_portion
    train_data = data[:train_portion]
    test_data = data[train_portion:train_portion+test_portion]
    valid_data = data[train_portion+test_portion:]
    logger.info(f"Training data length: {len(train_data)}")
    logger.info(f"Test data length: {len(test_data)}")
    logger.info(f"Validation data length: {len(valid_data)}")

if __name__ == "__main__":
    main()
