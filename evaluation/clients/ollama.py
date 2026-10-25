# -*- coding: utf-8 -*-

# ***************************************************
# * File        : ollama.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-02-17
# * Version     : 1.0.102511
# * Description : Ollama 客户端封装：check_if_running / query_model
# *               （/api/chat 端点）与 query_generate（/api/generate 端点，
# *               非流式，judge 等场景使用）。
# * Link        : link
# * Requirement : psutil
# ***************************************************

# python libraries
import os
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)
import json
import psutil
import urllib.error
import urllib.request


from utils.log_util import logger

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


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


def query_generate(prompt, model="gpt-oss:20b",
                   url="http://localhost:11434/api/generate",
                   timeout=60, seed=123, temperature=0.0, max_new_tokens=256):
    """调用本地 Ollama generate 接口（非流式，reasoning chF/04 judge 适配版）。

    与 query_model（/api/chat）不同，本函数走 /api/generate 端点，
    以 urllib 非流式 JSON 调用（标准库实现、无新增依赖）。

    Args:
        prompt: 完整 prompt 文本
        model: Ollama 模型名（judge 默认 gpt-oss:20b）
        url: Ollama 服务地址
        timeout: 请求超时（秒）
        seed: 采样种子（judge 期望确定性输出，配合 temperature=0）
        temperature: 采样温度
        max_new_tokens: 生成的最大 token 数

    Returns:
        str: 模型生成的回复文本

    Raises:
        RuntimeError: 连接失败（服务未启动/地址错误）等，附排查建议
    """
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "seed": seed,
            "temperature": temperature,
            "num_predict": int(max_new_tokens),
        },
    }).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, TimeoutError) as e:
        raise RuntimeError(
            f"无法连接 Ollama 服务（{url}）：{e}。"
            "请确认已安装并启动 Ollama（`ollama serve`），"
            f"且已拉取 judge 模型（`ollama pull {model}`）。"
        ) from e

    return data.get("response", "")


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
        raise RuntimeError("Ollama not running. Launch ollama before proceeding")

if __name__ == "__main__":
    main()
