# -*- coding: utf-8 -*-

# ***************************************************
# * File        : openai_api.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-03-25
# * Version     : 1.0.032523
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# * TODO        : 1.
# ***************************************************

__all__ = []

# python libraries
import os
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)
import json


from openai import OpenAI

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


def create_client():
    """
    openai client
    """
    # Load API key from a JSON file.
    with open("config.json", "r") as config_file:
        config = json.load(config_file)
        api_key = config["OPENAI_API_KEY"]
    # OpenAI client
    client = OpenAI(api_key=api_key)

    return client


def run_chatgpt(prompt, client, model="gpt-4-turbo"):
    """
    Run ChatGPT

    Args:
        prompt (_type_): _description_
        client (_type_): _description_
        model (str, optional): _description_. Defaults to "gpt-4-turbo".

    Returns:
        _type_: _description_
    """
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        seed=123,
    )
    return response.choices[0].message.content




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
