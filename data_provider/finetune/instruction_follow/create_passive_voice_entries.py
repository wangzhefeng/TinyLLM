# -*- coding: utf-8 -*-

# ***************************************************
# * File        : create_paasive_voice_entries.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-03-25
# * Version     : 1.0.032523
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

from tqdm import tqdm

from data_provider.load_save_data import load_json_data, save_json_data
from model_inference.inference_utils.openai_api import create_client, run_chatgpt

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]




# 测试代码 main 函数
def main():
    # instruction data path
    data_path="./dataset/finetune/instruction-example.json"

    # load instruction data
    json_data = load_json_data(data_path)

    # openai client
    client = create_client()
    
    # data process
    for i, entry in tqdm(enumerate(json_data), total=len(json_data)):
        text = entry["output"]
        prompt = f"Without adding any response or explanation, convert the following text to passive voice: {text}"
        json_data[i]["output_2"] = run_chatgpt(prompt, client)

    # save instruction data with passive voice output
    new_data_path = data_path.replace(".json", "-modified.json")
    save_json_data(json_data=json_data, save_path=new_data_path)

if __name__ == "__main__":
    main()
