# -*- coding: utf-8 -*-

# ***************************************************
# * File        : llm_judges.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-10-05
# * Version     : 2.0.102101
# * Description : LLM-as-a-Judge 核心逻辑（reasoning-from-scratch chF/04）：
# *               1-5 分 rubric 打分 prompt 构建、回复打分抽取、分数汇总。
# *               纯函数式实现，不依赖任何外部 API；Ollama/transformers
# *               等后端由调用方注入 query_fn（见 judges/ollama.py
# *               与 judges/qwen3.py）。
# * Link        : https://github.com/rasbt/reasoning-from-scratch/tree/main/chF/04_llm-judge
# * Requirement : 无（纯 Python）
# ***************************************************

__all__ = [
    "build_judge_prompt",
    "parse_score",
    "summarize_scores",
    "evaluate_response",
]

# python libraries
import os
import re
import statistics
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)
import warnings
warnings.filterwarnings("ignore")

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]
os.environ['LOG_NAME'] = LOGGING_LABEL
from utils.log_util import logger


def build_judge_prompt(question, response, reference_answer=None):
    """构建书中风格的 rubric 打分 prompt（1-5 分制）。

    书 chF/04 的 rubric_prompt 原文移植：judge 按 1-5 分的明确标准给候选
    回复打分；提供 reference_answer 时进入 "instruction + reference + answer"
    三段式（书原版），否则退化为 "instruction + answer" 两段式。

    Args:
        question: 指令/问题原文
        response: 待评的候选回复
        reference_answer: 参考答案（可选，有则评分更稳）

    Returns:
        str: 完整 judge prompt（以 "Evaluation: " 结尾，诱导模型输出评分）
    """
    rubric = (
        "You are a fair judge assistant. You will be given an instruction, "
        "a reference answer, and a candidate answer to evaluate, according "
        "to the following rubric:\n\n"
        "1: The response fails to address the instruction, providing "
        "irrelevant, incorrect, or excessively verbose content.\n"
        "2: The response partially addresses the instruction but contains "
        "major errors, omissions, or irrelevant details.\n"
        "3: The response addresses the instruction to some degree but is "
        "incomplete, partially correct, or unclear in places.\n"
        "4: The response mostly adheres to the instruction, with only minor "
        "errors, omissions, or lack of clarity.\n"
        "5: The response fully adheres to the instruction, providing a "
        "clear, accurate, and relevant answer in a concise and efficient "
        "manner.\n\n"
        "Now here is the instruction, the reference answer, and the "
        "response.\n"
    )

    parts = [f"{rubric}", f"Instruction:\n{question}\n"]
    if reference_answer is not None:
        parts.append(f"\nReference Answer:\n{reference_answer}\n")
    parts.append(f"\nAnswer:\n{response}\n\nEvaluation: ")

    return "".join(parts)


def parse_score(judge_text, default=3):
    """从 judge 回复中抽取 1-5 的整数评分（书 chF/04 原版正则）。

    正则 ``([1-5])(?:\\D|$)`` 取第一个后跟非数字/行尾的 1-5 数字，
    兼容 "Rating: 4"、"rating: 4"、"4/5"、"The score is 5." 等写法；
    无匹配时返回 default（书原版 default=3，避免极端惩罚格式抖动）。
    """
    m = re.search(r"([1-5])(?:\D|$)", judge_text)
    return int(m.group(1)) if m else int(default)


def summarize_scores(scores):
    """汇总一批评分：均值、中位数、各档计数。

    Args:
        scores: 评分数值迭代器（理想为 1-5）

    Returns:
        dict: {"mean", "median", "counts": {1: n1, ..., 5: n5}, "total"}
    """
    scores = list(scores)
    counts = {i: 0 for i in range(1, 6)}
    for s in scores:
        counts[int(s)] = counts.get(int(s), 0) + 1
    return {
        "mean": statistics.mean(scores) if scores else 0.0,
        "median": statistics.median(scores) if scores else 0.0,
        "counts": counts,
        "total": len(scores),
    }


def evaluate_response(question, response, query_fn, reference_answer=None,
                      default=3):
    """单条回复的完整 judge 流程：构建 prompt -> 调用后端 -> 抽分。

    Args:
        question: 指令/问题原文
        response: 待评回复
        query_fn: callable[prompt:str] -> judge 回复文本的后端函数，
            例如 evaluation.clients.ollama.query_generate；None 时抛出明确错误，
            提示必须注入后端（本模块自身不绑定任何 API）
        reference_answer: 参考答案（可选）
        default: parse_score 的兜底分

    Returns:
        dict: {"prompt", "judge_text", "score"}
    """
    if query_fn is None:
        raise ValueError(
            "query_fn 不能为 None：请注入一个 callable(prompt) -> str 的后端，"
            "例如 evaluation.clients.ollama.query_generate。"
        )

    prompt = build_judge_prompt(question, response, reference_answer)
    judge_text = query_fn(prompt)
    score = parse_score(judge_text, default=default)
    return {"prompt": prompt, "judge_text": judge_text, "score": score}



# 测试代码 main 函数
def main():
    """离线演示：mock query_fn（固定返回 "Rating: 4"）跑通 judge 全流程。"""
    mock_query = lambda prompt: "The response is mostly correct.\nRating: 4"  # noqa: E731

    result = evaluate_response(
        question="What is 2 + 2?",
        response="2 + 2 = 5",
        reference_answer="4",
        query_fn=mock_query,
    )
    logger.info(f"judge score: {result['score']}")
    logger.info(f"summary: {summarize_scores([result['score'], 5, 3, 4])}")

    # parse_score 兼容性演示
    for text in ["Rating: 4", "rating: 4", "4/5", "The score is 5.", "no digit here"]:
        logger.info(f"parse_score({text!r}) = {parse_score(text)}")


if __name__ == "__main__":
    main()
