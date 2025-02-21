# -*- coding: utf-8 -*-

# ***************************************************
# * File        : evaluating_instruction_flow.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2025-02-18
# * Version     : 0.1.021800
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
import psutil
from tqdm import tqdm
import urllib.request

from finetuning.instruction_follow.data_load import load_file
from utils.log_util import logger

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


def query_model(prompt, model="llama3", url="http://localhost:11434/api/chat"):
    # Create the data payload as a dictionary
    data = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "options": {     # Settings below are required for deterministic responses
            "seed": 123,
            "temperature": 0,
            "num_ctx": 2048
        }
    }

    # Convert the dictionary to a JSON formatted string and encode it to bytes
    payload = json.dumps(data).encode("utf-8")
    # Create a request object, setting the method to POST and adding necessary headers
    request = urllib.request.Request(url, data=payload, method="POST")
    request.add_header("Content-Type", "application/json")
    # Send the request and capture the response
    response_data = ""
    with urllib.request.urlopen(request) as response:
        # Read and decode the response
        while True:
            line = response.readline().decode("utf-8")
            if not line:
                break
            response_json = json.loads(line)
            response_data += response_json["message"]["content"]

    return response_data


def check_if_running(process_name):
    running = False
    for proc in psutil.process_iter(["name"]):
        if process_name in proc.info["name"]:
            running = True
            break
    
    return running


def format_input(entry):
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )

    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""

    return instruction_text + input_text


def generate_model_scores(json_data, json_key, model="llama3"):
    scores = []
    for entry in tqdm(json_data, desc="Scoring entries"):
        if entry[json_key] == "":
            scores.append(0)
        else:
            prompt = (
                f"Given the input `{format_input(entry)}` "
                f"and correct output `{entry['output']}`, "
                f"score the model response `{entry[json_key]}`"
                f" on a scale from 0 to 100, where 100 is the best score. "
                f"Respond with the integer number only."
            )
            score = query_model(prompt, model)
            try:
                scores.append(int(score))
            except ValueError:
                print(f"Could not convert score: {score}")
                continue

    return scores




# 测试代码 main 函数
def main():
    # data
    data = load_file(file_path = "./dataset/finetuning/instruction-data.json")

    # data split
    train_portion = int(len(data) * 0.85)
    test_portion = int(len(data) * 0.10)
    valid_portion = len(data) - train_portion - test_portion
    train_data = data[:train_portion]
    test_data = data[train_portion:train_portion+test_portion]
    valid_data = data[train_portion+test_portion:]
    # logger.info(f"Training data length: {len(train_data)}")
    # logger.info(f"Test data length: {len(test_data)}")
    # logger.info(f"Validation data length: {len(valid_data)}")
    # ------------------------------
    # command
    # ------------------------------
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate model responses with ollama")
    parser.add_argument("--file_path", required=True,
                        help="The path to the test dataset `.json` file with the `'output'` and `'model_response'` keys")
    args = parser.parse_args()
    # ------------------------------
    # 
    # ------------------------------
    ollama_running = check_if_running("ollama")

    if not ollama_running:
        raise RuntimeError("Ollama not running. Launch ollama before proceeding.")
    print("Ollama running:", check_if_running("ollama"))

    with open(args.file_path, "r") as file:
        test_data = json.load(file)

    model = "llama3"
    scores = generate_model_scores(test_data, "model_response", model)
    print(f"Number of scores: {len(scores)} of {len(test_data)}")
    print(f"Average score: {sum(scores)/len(scores):.2f}\n")

if __name__ == "__main__":
    main()
