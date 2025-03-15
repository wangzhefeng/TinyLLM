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

from utils.log_util import logger

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


def _format_input(entry):
    """
    format instruction input
    """
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )

    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""

    return instruction_text + input_text


def check_if_running(process_name = "ollama"):
    """
    check if ollama is running
    """
    running = False
    for proc in psutil.process_iter(["name"]):
        if process_name in proc.info["name"]:
            running = True
            break
    
    return running


def _query_model(prompt, model="llama3", url="http://localhost:11434/api/chat", seed=123, num_ctx=2048):
    """
    query ollama REST API in Python

    Args:
        prompt (_type_): _description_
        model (str, optional): _description_. Defaults to "llama3".
        url (str, optional): _description_. Defaults to "http://localhost:11434/api/chat".

    Returns:
        _type_: _description_
    """ 
    # Create the data payload as a dictionary
    data = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        # Settings below are required for deterministic responses
        "options": {
            "seed": seed,
            "temperature": 0,
            "num_ctx": num_ctx,
        }
    }
    # Convert the dictionary to a JSON formatted string and encode it to bytes
    payload = json.dumps(data).encode("utf-8")
    # Create a request object, setting the method to POST and adding necessary headers
    request = urllib.request.Request(url, data = payload, method = "POST")
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


def generate_model_scores(
    json_data, 
    json_key, 
    model = "llama3", 
    url = "http://localhost:11434/api/chat", 
    seed=123, 
    num_ctx=2048
    ):
    """
    generate model evaluate scores
    """
    scores = []
    for entry in tqdm(json_data, desc="Scoring entries"):
        if entry[json_key] == "":
            scores.append(0)
        else:
            # query model to generate score
            prompt = (
                f"Given the input `{_format_input(entry)}` "
                f"and correct output `{entry['output']}`, "
                f"score the model response `{entry[json_key]}`"
                f" on a scale from 0 to 100, where 100 is the best score. "
                f"Respond with the integer number only."
            )
            score = _query_model(
                prompt=prompt, 
                model=model, 
                url=url, 
                seed=seed, 
                num_ctx=num_ctx
            )
            # print the procession
            logger.info(f"\nDataset response:")
            logger.info(f">>, {entry['output']}")
            logger.info(f"\nModel response:")
            logger.info(f">>, {entry['model_response']}")
            logger.info(f"\nScore:")
            logger.info(f">>, {score}")
            logger.info(f"\n-------------------------")

            # save scores
            try:
                scores.append(int(score))
            except ValueError:
                logger.info(f"Could not convert score: {score}")
                continue

    return scores




# 测试代码 main 函数
def main(): 
    model = "llama3"
    inference_server = "ollama"

    # check
    ollama_running = check_if_running(process_name=inference_server)
    if not ollama_running:
        raise RuntimeError("Ollama not running. Launch ollama before proceeding.")
    logger.info(f"Ollama running: {ollama_running}")

    # query model
    result = _query_model("What do Llamas eat?", model)
    logger.info(f"result: \n{result}")

if __name__ == "__main__":
    main()
