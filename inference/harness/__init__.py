# -*- coding: utf-8 -*-

# ***************************************************
# * File        : __init__.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2026-10-25
# * Version     : 1.0.102511
# * Description : MATH-500 端到端评测 harness（生成->解析->评分->统计）
# *               与多项选择基准（mmlu）。沿用原 harness 的导出；
# *               CLI 见 python -m inference.harness.math500 --help
# * Requirement : sympy（经 evaluation.verifiers.math）；CLI 另需 torch
# ***************************************************

__all__ = [
    "evaluate_math500",
    "evaluate_math500_batched",
    "save_report",
    "heuristic_score",
]

from inference.harness.math500 import (
    evaluate_math500,
    evaluate_math500_batched,
    save_report,
    heuristic_score,
)
