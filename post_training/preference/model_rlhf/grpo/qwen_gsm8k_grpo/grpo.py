# -*- coding: utf-8 -*-

# ***************************************************
# * File        : reward_model.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-07-20
# * Version     : 1.0.072017
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

__all__ = []

# python libraries
import os
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)
import re
from typing import List
import warnings
warnings.filterwarnings("ignore")

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]
os.environ['LOG_NAME'] = LOGGING_LABEL
from utils.log_util import logger


def extract_xml_answer(text: str) -> str:
    """
    extract xml answer
    """
    match = re.search("<answer>(.*)</answer>", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        return ""


def correctness_reward_func(prompts, completions, answer, **kwargs) -> List[float]:
    """
    TODO
    """
    responses = [c[0]["content"] for c in completions]
    extracted = [extract_xml_answer(r) for r in responses]
    rewards = []
    for r, a in zip(extracted, answer):
        if a in r:
            reward = 1.0
        else:
            reward = 0.0
        rewards.append(reward)

    return rewards


def soft_format_reward_func(completions, **kwargs) -> List[float]:
    """
    TODO
    """
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [c[0]["content"] for c in completions]
    rewards = []
    for r in responses:
        if re.search(pattern, r, re.DOTALL):
            reward = 2.0
        else:
            reward = 0.0
        rewards.append(reward)

    return rewards


def strict_format_reward_func(completions, **kwargs) -> List[float]:
    """
    TODO
    """
    pattern = r"^\s*<reasoning>.*?</reasoning>\s*<answer>.*?</answer>\s*$"
    responses = [c[0]["content"] for c in completions]
    rewards = []
    for r in responses:
        if re.search(pattern, r, re.DOTALL):
            reward = 4.0
        else:
            reward = 0.0
        rewards.append(reward)

    return rewards




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
