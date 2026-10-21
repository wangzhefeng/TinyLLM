# -*- coding: utf-8 -*-

# ***************************************************
# * File        : llm_judges_Qwen3.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-10-12
# * Version     : 2.0.102101
# * Description : LLM-as-a-Judge 的 Qwen3 本地后端（reasoning-from-scratch
# *               chF/04 主题的本地化变体）：judge 模型不依赖外部服务，
# *               用 transformers pipeline 加载本地 Qwen3；transformers
# *               延迟到函数内 import，模块级零重依赖。纯打分逻辑复用
# *               llm_judges.py。
# * Link        : https://github.com/rasbt/reasoning-from-scratch/tree/main/chF/04_llm-judge
# * Requirement : transformers（仅调用 query_qwen3_pipeline 时需要）
# ***************************************************

__all__ = [
    "query_qwen3_pipeline",
    "evaluate_response_qwen3",
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

from post_training.reasoning.evaluation.judgement_based_eval.llm_judges import (
    evaluate_response,
)

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]
os.environ['LOG_NAME'] = LOGGING_LABEL
from utils.log_util import logger

# pipeline 懒加载单例（首次调用 query_qwen3_pipeline 时创建）
_QWEN3_PIPELINE = None


def query_qwen3_pipeline(prompt, model_name="Qwen/Qwen3-0.6B",
                         max_new_tokens=256, temperature=0.0,
                         device=None, cache_dir="./downloaded_models/"):
    """用 transformers pipeline 加载本地 Qwen3 作为 judge 后端。

    transformers 在函数内延迟 import（避免模块级重依赖）；首次调用会
    下载/加载模型并缓存 pipeline（模块级全局变量复用），后续调用零开销。

    Args:
        prompt: 完整 prompt 文本
        model_name: judge 模型名（默认 Qwen3-0.6B，教学用小模型）
        max_new_tokens: 最大生成 token 数
        temperature: 采样温度（judge 期望确定性输出，默认 0）
        device: torch 设备；None 时自动选择（cuda > mps > cpu）
        cache_dir: 模型缓存目录

    Returns:
        str: 模型生成的回复文本
    """
    # 延迟 import：仅实际调用 judge 时才需要 transformers
    import torch
    from transformers import pipeline as hf_pipeline

    # pipeline 缓存（模块级懒加载单例）
    global _QWEN3_PIPELINE
    if _QWEN3_PIPELINE is None:
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        logger.info(f"loading judge model {model_name} on {device} ...")
        _QWEN3_PIPELINE = hf_pipeline(
            "text-generation",
            model=model_name,
            device=device,
            cache_dir=cache_dir,
        )

    outputs = _QWEN3_PIPELINE(
        prompt,
        max_new_tokens=int(max_new_tokens),
        temperature=temperature,
        do_sample=temperature > 0.0,
        return_full_text=False,          # 只返回生成部分
    )
    return outputs[0]["generated_text"]


def evaluate_response_qwen3(question, response, reference_answer=None,
                            model_name="Qwen/Qwen3-0.6B", **query_kwargs):
    """Qwen3 本地后端的完整 judge 流程（llm_judges.evaluate_response 的封装）。"""
    return evaluate_response(
        question=question,
        response=response,
        query_fn=lambda prompt: query_qwen3_pipeline(
            prompt, model_name=model_name, **query_kwargs
        ),
        reference_answer=reference_answer,
    )



# 测试代码 main 函数
def main():
    """离线演示：mock query_fn 跑通纯逻辑；真实调用见 evaluate_response_qwen3。"""
    from post_training.reasoning.evaluation.judgement_based_eval.llm_judges import (
        build_judge_prompt,
        summarize_scores,
    )

    prompt = build_judge_prompt(
        question="What is the capital of France?",
        response="The capital of France is Paris.",
        reference_answer="Paris",
    )
    logger.info(f"judge prompt 预览:\n{prompt[:300]}...")
    mock_query = lambda p: "Rating: 5"  # noqa: E731
    result = evaluate_response(
        question="What is the capital of France?",
        response="The capital of France is Paris.",
        reference_answer="Paris",
        query_fn=mock_query,
    )
    logger.info(f"mock judge score: {result['score']}")
    logger.info(f"summary: {summarize_scores([result['score']])}")
    logger.info("真实 judge：evaluate_response_qwen3(question, response)")


if __name__ == "__main__":
    main()
