# -*- coding: utf-8 -*-

# ***************************************************
# * File        : ollama.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-10-12
# * Version     : 2.0.102511
# * Description : LLM-as-a-Judge 的 Ollama 后端（reasoning-from-scratch
# *               chF/04 的 ollama-judge.py 适配）：generate 端点调用
# *               复用 evaluation.clients.ollama.query_generate（非流式
# *               JSON 调本地 Ollama /api/generate），组合
# *               llm_judges.evaluate_response 形成完整 judge 流程。
# *               模块级零调用、零网络；本地无 Ollama 时给出清晰报错。
# * Link        : https://github.com/rasbt/reasoning-from-scratch/tree/main/chF/04_llm-judge
# * Requirement : 无（urllib 为标准库，在 clients.ollama 内使用）
# ***************************************************

__all__ = [
    "evaluate_response_ollama",
]

# python libraries
import os
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)
import warnings
warnings.filterwarnings("ignore")

from evaluation.clients.ollama import query_generate
from evaluation.judges.llm_judges import (
    evaluate_response,
)

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]
os.environ['LOG_NAME'] = LOGGING_LABEL
from utils.log_util import logger


def evaluate_response_ollama(question, response, reference_answer=None,
                             model="gpt-oss:20b",
                             url="http://localhost:11434/api/generate",
                             **query_kwargs):
    """Ollama 后端的完整 judge 流程（llm_judges.evaluate_response 的封装）。"""
    return evaluate_response(
        question=question,
        response=response,
        query_fn=lambda prompt: query_generate(
            prompt, model=model, url=url, **query_kwargs
        ),
        reference_answer=reference_answer,
    )



# 测试代码 main 函数
def main():
    """离线演示：只打印 judge prompt 构建结果；实际调用需本地 Ollama。"""
    from evaluation.judges.llm_judges import (
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
