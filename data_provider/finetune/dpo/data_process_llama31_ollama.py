# -*- coding: utf-8 -*-

# ***************************************************
# * File        : preference_data_llama3170B_ollama.py
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
from tqdm import tqdm

from data_provider.finetune.instruction_follow.instruction_format import format_input_alpaca
from utils.inference_utils.ollama_api import query_model
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


def generate_model_response(json_data):
    """
    将提示（prompt）应用于 整个数据集。
    在数据集中添加：
        - 'chosen'：代表 偏好（preferred）响应
        - 'rejected'：代表 非偏好（dispreferred）响应

    Args:
        json_data (_type_): _description_

    Returns:
        _type_: _description_
    """
    for i, entry in enumerate(tqdm(json_data, desc = "Writing entries")):
        politeness = random.choice(["polite", "impolite"])
        prompt = (
            f"Given the input `{format_input_alpaca(entry)}` "
            f"and correct output `{entry['output']}`, "
            f"slightly rewrite the output to be more {politeness}."
            "Keep the modification minimal."
            "Only return return the generated response and nothing else."
        )
        response = query_model(prompt, model='llama3.1')

        if politeness == "polite":
            json_data[i]["chosen"] = response
            json_data[i]["rejected"] = entry["output"]
        else:
            json_data[i]["rejected"] = response
            json_data[i]["chosen"] = entry["output"]

    return json_data


def save_instruction_with_preference(json_data, save_path: str):
    """
    save instruction entries with preference json data
    """
    with open(save_path, "w") as file:
        json.dump(json_data, file, indent=4)




# 测试代码 main 函数
def main():
    instruction_entries_path = "./dataset/finetune/instruction-data.json"
    instruction_entries_with_preference_path = "./dataset/finetune/instruction-preference-data.json"
    instruction_entries_data = load_instruction_data(
        data_path = instruction_entries_path
    )
    logger.info(f"Number of entries: {len(instruction_entries_data)}")

    if not os.path.exists(instruction_entries_with_preference_path):
        # generate model response
        instruction_entries_with_preference_data = generate_model_response(
            json_data = instruction_entries_data
        )
        logger.info(f"Number of entries: {len(instruction_entries_with_preference_data)}")
        # save instruction entries with preference
        save_instruction_with_preference(
            instruction_entries_data, 
            save_path = instruction_entries_with_preference_path,
        )

if __name__ == "__main__":
    main()
