# -*- coding: utf-8 -*-

# ***************************************************
# * File        : load_entries.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2025-02-22
# * Version     : 0.1.022200
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
import random

from finetuning.instruction_follow.data_process import format_input_alpaca
from inference_local.ollama_api import query_model
from utils.log_util import logger

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


# instruction entries json data
instruction_entries_path = "./dataset/finetuning/instruction-data.json"
with open(instruction_entries_path, "r") as file:
    instruction_entries_data = json.load(file)
logger.info(f"Number of entries: {len(instruction_entries_data)}")

# perference finetuning
for entry in instruction_entries_data[:5]:
    politeness = random.choice(["polite", "impolite"])
    prompt = (
        f"Given the input `{format_input_alpaca(entry)}` "
        f"and correct output `{entry['output']}`, "
        f"slightly rewrite the output to be more {politeness}."
        "Keep the modification minimal."
        "Only return return the generated response and nothing else."
    )
    logger.info(f"Dataset response:")
    logger.info(f">> {entry['output']}")
    logger.info(f"{politeness} response:")
    logger.info(f">> {query_model(prompt, model='llama3.1')}")





# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
