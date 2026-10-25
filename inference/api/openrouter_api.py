# -*- coding: utf-8 -*-

# ***************************************************
# * File        : openrouter_api.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2026-10-24
# * Version     : 1.0.102421
# * Description : OpenRouter API client 封装（openai SDK 兼容端点，
# *               reasoning-from-scratch ch08 蒸馏教师模型入口）。
# *               凭证从环境变量读取：OPENROUTER_API_KEY（必需）、
# *               OPENROUTER_BASE_URL（可选，默认官方 https://openrouter.ai/api/v1）。
# * Link        : https://github.com/rasbt/reasoning-from-scratch/tree/main/ch08/02_generate_distillation_data
# * Requirement : pip install openai
# ***************************************************

__all__ = [
    "create_client",
    "run_openrouter",
]

# python libraries
import os
import sys
from pathlib import Path
# 仓根路径由本文件位置推导（inference/api/openrouter_api.py -> 上两级），不依赖启动时的 cwd
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

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "deepseek/deepseek-r1"


def create_client():
    """
    openrouter client（openai SDK 兼容端点）

    从环境变量读取凭证：
      - OPENROUTER_API_KEY: 必需，未设置时抛出带指引的 RuntimeError
      - OPENROUTER_BASE_URL: 可选，默认 https://openrouter.ai/api/v1
    """
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "未设置 OPENROUTER_API_KEY 环境变量，无法创建 OpenRouter client。"
            "请先 export OPENROUTER_API_KEY=<your-key>。"
        )
    if OpenAI is None:
        raise RuntimeError("未安装 openai 包，请先 pip install openai")
    base_url = os.environ.get("OPENROUTER_BASE_URL", DEFAULT_BASE_URL)
    client = OpenAI(api_key=api_key, base_url=base_url)

    return client


def run_openrouter(prompt, client, model=DEFAULT_MODEL, max_tokens=None):
    """
    Run OpenRouter chat completion

    Args:
        prompt: 用户提示文本
        client: create_client() 返回的 openai SDK client
        model: OpenRouter 模型名，Defaults to "deepseek/deepseek-r1".
        max_tokens: 可选生成上限（蒸馏数据建议 2048，长思维链需要）；
            None 时不传该参数

    Returns:
        助手回复文本
    """
    kwargs = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
    }
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content



# 测试代码 main 函数
def main():
    client = create_client()
    result = run_openrouter("Reply with OK.", client)
    print(f"OpenRouter reply: {result}")

if __name__ == "__main__":
    main()
