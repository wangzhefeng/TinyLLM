# -*- coding: utf-8 -*-

# ***************************************************
# * File        : openai_api.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-03-25
# * Version     : 1.0.032523
# * Description : OpenAI API client 封装。
# *               凭证从环境变量读取：OPENAI_API_KEY（必需）、
# *               OPENAI_BASE_URL（可选，用于兼容代理/自托管端点）。
# * Link        : link
# * Requirement : pip install openai
# ***************************************************

__all__ = []

# python libraries
import os
import sys
from pathlib import Path
# 仓根路径由本文件位置推导（inference/api/openai_api.py -> 上两级），不依赖启动时的 cwd
ROOT = str(Path(__file__).resolve().parents[2])
if ROOT not in sys.path:
    sys.path.append(ROOT)

# openai 为可选依赖：未安装时模块仍可导入，create_client() 给出清晰报错
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


def create_client():
    """
    openai client

    从环境变量读取凭证：
      - OPENAI_API_KEY: 必需，未设置时抛出带指引的 RuntimeError
      - OPENAI_BASE_URL: 可选，自定义 API 端点（代理/自托管兼容服务）
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "未设置 OPENAI_API_KEY 环境变量，无法创建 OpenAI client。"
            "请先 export OPENAI_API_KEY=<your-key>（可选 OPENAI_BASE_URL 指定端点）。"
        )
    if OpenAI is None:
        raise RuntimeError("未安装 openai 包，请先 pip install openai")
    base_url = os.environ.get("OPENAI_BASE_URL")
    kwargs = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    client = OpenAI(**kwargs)

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
