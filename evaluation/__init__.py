# -*- coding: utf-8 -*-

# ***************************************************
# * File        : __init__.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2026-10-25
# * Version     : 1.0.102511
# * Description : evaluation 公共评测库（懒加载，import 本包无副作用）：
# *                 from evaluation import query_model, grade_answer, evaluate_response
# *               子包：clients（模型服务客户端）/ verifiers（答案解析与
# *               数学等价验证）/ judges（LLM-as-a-Judge 与排行榜）。
# *               注意：import 本包不拉起 torch；verifiers 依赖 sympy 属可接受。
# * Requirement : 无重依赖（懒加载）
# ***************************************************

__all__ = [
    # clients：本地/远程模型服务客户端
    "check_if_running",
    "query_model",
    "query_generate",
    "create_client",
    "run_chatgpt",
    "run_openrouter",
    # verifiers：答案解析与数学等价性验证
    "get_last_boxed",
    "extract_final_candidate",
    "grade_answer",
    "parse_and_verify_answer",
    "render_prompt",
    "load_math500_data",
    # judges：LLM-as-a-Judge 纯逻辑与后端
    "build_judge_prompt",
    "parse_score",
    "summarize_scores",
    "evaluate_response",
    "evaluate_response_ollama",
    "evaluate_response_qwen3",
    # judges：排行榜（Elo / Bradley-Terry）
    "elo_ratings",
    "expected_score",
    "update_elo",
    "format_leaderboard",
    "fit_bradley_terry",
]


def __getattr__(name):
    """PEP 562 懒加载：避免 import evaluation 时拉起 torch/transformers。"""
    # clients
    if name in ("check_if_running", "query_model", "query_generate"):
        from evaluation.clients import ollama
        return getattr(ollama, name)
    if name in ("create_client", "run_chatgpt"):
        from evaluation.clients import openai
        return getattr(openai, name)
    if name == "run_openrouter":
        from evaluation.clients import openrouter
        return getattr(openrouter, name)
    # verifiers
    if name in (
        "get_last_boxed",
        "extract_final_candidate",
        "grade_answer",
        "render_prompt",
        "load_math500_data",
    ):
        from evaluation.verifiers import math
        return getattr(math, name)
    if name == "parse_and_verify_answer":
        from evaluation.verifiers import parser
        return getattr(parser, name)
    # judges：LLM-as-a-Judge
    if name in ("build_judge_prompt", "parse_score", "summarize_scores", "evaluate_response"):
        from evaluation.judges import llm_judges
        return getattr(llm_judges, name)
    if name == "evaluate_response_ollama":
        from evaluation.judges import ollama
        return getattr(ollama, name)
    if name == "evaluate_response_qwen3":
        from evaluation.judges import qwen3
        return getattr(qwen3, name)
    # judges：排行榜
    if name in ("elo_ratings", "expected_score", "update_elo", "format_leaderboard",
                "fit_bradley_terry"):
        from evaluation.judges import leaderboards
        return getattr(leaderboards, name)
    raise AttributeError(f"module 'evaluation' has no attribute {name!r}")
