# -*- coding: utf-8 -*-

# ***************************************************
# * File        : instruction_format.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-02-23
# * Version     : 0.1.022300
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


# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


def format_input_alpaca(entry):
    """
    format the input to the LLM use Alpaca-style prompt formatting
    
    Alpaca: https://crfm.stanford.edu/2023/03/13/alpaca.html
    """
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )
    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""
    # response_text = f"\n\n### Response:\n{entry['output']}"

    return instruction_text + input_text #+ response_text


# TODO
def format_input_phi3(entry):
    """
    Phi-3 paper:https://arxiv.org/abs/2404.14219
    """
    instruction_text = f"<|user|>\n{entry['instruction']}"
    input_text = f": '{entry['input']}'"
    # response_text = f"<|assistant|>\n{entry["output"]}"

    return instruction_text[:-1] + input_text




# 测试代码 main 函数
def main():
    from data_provider.finetune.instruction_follow import data_config
    from data_provider.load_save_data import load_json_data
    from utils.log_util import logger

    # data load
    data = load_json_data(data_path = data_config.data_path)

    # prompt format
    formated_entry = format_input_alpaca(data[0])
    logger.info(f"format instruction entry: \n{formated_entry}")

    # prompt format
    formated_entry = format_input_alpaca(data[50])
    logger.info(f"format instruction entry: \n{formated_entry}")

    # prompt format
    formated_entry = format_input_alpaca(data[999])
    logger.info(f"format instruction entry: \n{formated_entry}")

    # prompt format
    formated_entry = format_input_phi3(data[50])
    logger.info(f"format instruction entry: \n{formated_entry}")

    # prompt format
    formated_entry = format_input_phi3(data[999])
    logger.info(f"format instruction entry: \n{formated_entry}")

if __name__ == "__main__":
    main()
