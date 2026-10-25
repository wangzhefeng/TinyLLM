# -*- coding: utf-8 -*-

# ***************************************************
# * File        : __init__.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2026-10-24
# * Version     : 1.0.102421
# * Description : MATH-500 端到端评测 harness（生成->解析->评分->统计）。
# *               生成函数可注入替身，评测循环完全离线可单测；
# *               CLI 见 python -m inference.evaluation.harness.math500 --help
# * Requirement : sympy（经 math_verifiers）；CLI 另需 torch
# ***************************************************

__all__ = [
    "evaluate_math500",
    "evaluate_math500_batched",
    "save_report",
    "heuristic_score",
]

from inference.evaluation.harness.math500 import (
    evaluate_math500,
    evaluate_math500_batched,
    save_report,
    heuristic_score,
)
