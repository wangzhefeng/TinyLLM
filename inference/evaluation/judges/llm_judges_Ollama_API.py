# -*- coding: utf-8 -*-

# ***************************************************
# * File        : llm_judges_Ollama_API.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-10-12
# * Version     : 2.0.102101
# * Description : LLM-as-a-Judge 的 Ollama 后端（reasoning-from-scratch
# *               chF/04 的 ollama-judge.py 适配）：query_ollama 用 urllib
# *               以非流式 JSON 调用本地 Ollama /api/generate，组合
# *               llm_judges.evaluate_response 形成完整 judge 流程。
# *               模块级零调用、零网络；本地无 Ollama 时给出清晰报错。
# * Link        : https://github.com/rasbt/reasoning-from-scratch/tree/main/chF/04_llm-judge
# * Requirement : 无（urllib 为标准库）
# ***************************************************

__all__ = [
    "query_ollama",
    "evaluate_response_ollama",
]

# python libraries
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)
import warnings
warnings.filterwarnings("ignore")

from inference.evaluation.judges.llm_judges import (
    evaluate_response,
)

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]
os.environ['LOG_NAME'] = LOGGING_LABEL
from utils.log_util import logger


def query_ollama(prompt, model="gpt-oss:20b",
                 url="http://localhost:11434/api/generate",
                 timeout=60, seed=123, temperature=0.0, max_new_tokens=256):
    """调用本地 Ollama generate 接口（非流式，书 chF/04 的简化版）。

    书原版用 requests 流式逐行读 /api/chat；此处按教学需要简化为
    urllib 非流式 /api/generate（标准库实现、无新增依赖）。

    Args:
        prompt: 完整 prompt 文本
        model: Ollama 模型名（judge 默认 gpt-oss:20b，与书一致）
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


def evaluate_response_ollama(question, response, reference_answer=None,
                             model="gpt-oss:20b",
                             url="http://localhost:11434/api/generate",
                             **query_kwargs):
    """Ollama 后端的完整 judge 流程（llm_judges.evaluate_response 的封装）。"""
    return evaluate_response(
        question=question,
        response=response,
        query_fn=lambda prompt: query_ollama(
            prompt, model=model, url=url, **query_kwargs
        ),
        reference_answer=reference_answer,
    )



# 测试代码 main 函数
def main():
    """离线演示：只打印 judge prompt 构建结果；实际调用需本地 Ollama。"""
    from inference.evaluation.judges.llm_judges import (
        build_judge_prompt,
        parse_score,
    )

    prompt = build_judge_prompt(
        question="What is 2 + 2?",
        response="2 + 2 = 4",
        reference_answer="4",
    )
    logger.info(f"judge prompt 预览:\n{prompt[:300]}...")
    logger.info(
        "启动 Ollama 后可运行：evaluate_response_ollama(question, response)"
        f"（示例抽分：parse_score('Rating: 4') = {parse_score('Rating: 4')}）"
    )


if __name__ == "__main__":
    main()
