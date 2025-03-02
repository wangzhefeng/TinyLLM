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

from openai import OpenAI

from utils.log_util import logger

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


# ------------------------------
# openai client
# ------------------------------
# Load API key from a JSON file.
with open("config.json", "r") as config_file:
    config = json.load(config_file)
    api_key = config["OPENAI_API_KEY"]

client = OpenAI(api_key=api_key)

# ------------------------------
# load json entries
# ------------------------------
json_file = "eval-example-data.json"

with open(json_file, "r") as file:
    json_data = json.load(file)

print("Number of entries:", len(json_data))
logger.info(f"json_data[0]:{json_data[0]}")


def format_input(entry):
    instruction_text = (
        f"Below is an instruction that describes a task. Write a response that "
        f"appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )

    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""
    instruction_text + input_text

    return instruction_text + input_text


def run_chatgpt(prompt, client, model="gpt-4-turbo"):
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        seed=123,
    )
    return response.choices[0].message.content


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
