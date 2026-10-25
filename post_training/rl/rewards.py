# -*- coding: utf-8 -*-

# ***************************************************
# * File        : rewards.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2026-10-24
# * Version     : 1.0.102421
# * Description : RLVR reward 函数（reasoning-from-scratch ch06/ch07）。
# *               correct_answer_reward：答案正确性主奖励（verifier 打分）；
# *               format_reward：<think>...</think> 与 \boxed{} 格式奖励。
# *               重依赖（sympy/evaluation.verifiers.math）在调用时懒加载。
# * Link        : https://github.com/rasbt/reasoning-from-scratch/blob/main/ch07/03_rlvr_grpo_scripts_advanced/7_6_plus_format_reward.py
# * Requirement : sympy（经 evaluation.verifiers.math）
# ***************************************************

__all__ = [
    "correct_answer_reward",
    "format_reward",
]

# python libraries
import re
from pathlib import Path

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_BOXED_RE = re.compile(r"\\boxed\{")


def correct_answer_reward(extract_fn=None, grade_fn=None):
    """构造答案正确性 reward（RLVR 主奖励）。

    内部懒加载 ``evaluation.verifiers.math`` 的
    ``extract_final_candidate`` / ``grade_answer``（reasoning ch03 verifier）。

    Args:
        extract_fn: 自定义抽取函数 ``fn(text, fallback=None) -> str``；
            None 时用 extract_final_candidate（要求 \\boxed{}，不做数字回退）
        grade_fn: 自定义判分函数 ``fn(pred, expected) -> bool``；
            None 时用 grade_answer（sympy 数学等价）

    Returns:
        reward_fn(generated_text, expected_answer) -> float（1.0 / 0.0）
    """

    def reward_fn(generated_text, expected_answer) -> float:
        from evaluation.verifiers.math import (
            extract_final_candidate,
            grade_answer,
        )

        extract = extract_fn if extract_fn is not None else extract_final_candidate
        grade = grade_fn if grade_fn is not None else grade_answer
        extracted = extract(generated_text, fallback=None)
        if not extracted:
            return 0.0
        return float(bool(grade(extracted, expected_answer)))

    return reward_fn


def format_reward(text: str, require_think: bool = True,
                  require_boxed: bool = True) -> float:
    """格式 reward：检查 ``<think>...</think>`` 闭合段与 ``\\boxed{}``。

    按满足项给分：得分 = 满足的要求项数 / 要求项数
    （默认两项都要求：都满足 1.0，仅一项 0.5，都不满足 0.0；
    require_think/require_boxed 均为 False 时无要求项，恒为 1.0）。

    原书 7_6 脚本在 token id 层面检查 think/end-think 出现顺序；
    本实现是文本层面的等价简化（闭合段内天然保证先 <think> 后 </think>）。
    """
    checks = []
    if require_think:
        checks.append(bool(_THINK_RE.search(text)))
    if require_boxed:
        checks.append(bool(_BOXED_RE.search(text)))
    if not checks:
        return 1.0
    return sum(checks) / len(checks)


# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
