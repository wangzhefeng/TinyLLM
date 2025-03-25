# -*- coding: utf-8 -*-

# ***************************************************
# * File        : evaluating_instruction_flow_openai.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2025-02-18
# * Version     : 0.1.021800
# * Description : https://github.com/rasbt/LLMs-from-scratch/blob/16738b61fd37bd929ea3b1982857608036d451fa/ch07/03_model-evaluation/llm-instruction-eval-openai.ipynb
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
from tqdm import tqdm
from pathlib import Path

from data_provider.load_save_data import load_json_data
from utils.inference_utils.openai_api import create_client, run_chatgpt
from utils.log_util import logger

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


def format_input(entry):
    instruction_text = (
        f"Below is an instruction that describes a task. Write a response that "
        f"appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )

    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""
    instruction_text + input_text

    return instruction_text + input_text


def generate_model_scores(json_data, json_key, client):
    scores = []
    for entry in tqdm(json_data, desc="Scoring entries"):
        prompt = (
            f"Given the input `{format_input(entry)}` "
            f"and correct output `{entry['output']}`, "
            f"score the model response `{entry[json_key]}`"
            f" on a scale from 0 to 100, where 100 is the best score. "
            f"Respond with the number only."
        )
        score = run_chatgpt(prompt, client)
        try:
            scores.append(int(score))
        except ValueError:
            continue

    return scores




# 测试代码 main 函数
def main():
    # instruction data path
    data_path="./dataset/finetune/instruction-example.json"

    # load instruction data
    json_data = load_json_data(data_path)

    # openai client
    client = create_client()
    
    for model in ("model 1 response", "model 2 response"):
        scores = generate_model_scores(json_data, model, client)
        print(f"\n{model}")
        print(f"Number of scores: {len(scores)} of {len(json_data)}")
        print(f"Average score: {sum(scores)/len(scores):.2f}\n")

        # Optionally save the scores
        save_path = Path("scores") / f"gpt4-{model.replace(' ', '-')}.json"
        with open(save_path, "w") as file:
            json.dump(scores, file)

if __name__ == "__main__":
    main()
