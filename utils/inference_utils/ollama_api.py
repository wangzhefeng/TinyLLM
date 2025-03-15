# -*- coding: utf-8 -*-

# ***************************************************
# * File        : llama3_model.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2025-02-17
# * Version     : 0.1.021723
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
import urllib.request

from utils.log_util import logger

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


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


def query_model(prompt, model="llama3.1:70b", url="http://localhost:11434/api/chat", seed=123, num_ctx=2048):
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
            {
                "role": "user", 
                "content": prompt
            }
        ],
        # Settings below are required for deterministic responses
        "options": {
            "seed": seed,
            "temperature": 0,
            "num_ctx": num_ctx,
        },
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




# 测试代码 main 函数
def main():
    model = "llama3"
    inference_server = "ollama"

    # check ollama running
    ollama_running = check_if_running(process_name=inference_server)
    # inference
    if ollama_running:
        logger.info(f"Ollama running: {ollama_running}")
        # query model
        result = query_model("What do Llamas eat?", model)
        logger.info(f"result: \n{result}")
    else:
        raise RuntimeError("Ollama not running. Launch ollama before proceeding.")

if __name__ == "__main__":
    main()
