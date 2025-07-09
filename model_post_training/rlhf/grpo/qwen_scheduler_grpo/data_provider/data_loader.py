# -*- coding: utf-8 -*-

# ***************************************************
# * File        : data_loader.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-05-05
# * Version     : 1.0.050500
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

__all__ = [
    "data_load"
]

# python libraries
import os
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)


from datasets import load_dataset

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]

from utils.log_util import logger


SYSTEM_PROMPT = """You are a precise event scheduler.
1. First, reason through the problem inside <think> and </think> tags. Here you can create drafts, compare alternatives, and check for mistakes.
2. When confident, output the final schedule inside <schedule> and </schedule> tags. Your schedule must strictly follow the rules provided by the user."""

USER_PROMPT = """Task: create an optimized schedule based on the given events.

Rules:
- The schedule MUST be in strict chronological order. Do NOT place priority events earlier unless their actual start time is earlier.
- Event start and end times are ABSOLUTE. NEVER change, shorten, adjust, or split them.
- Priority events (weight = 2) carry more weight than normal events (weight = 1), but they MUST still respect chronological order.
- Maximize the sum of weighted event durations.
- No overlaps allowed. In conflicts, include the event with the higher weighted time.
- Some events may be excluded if needed to meet these rules.


You must use this format:  

<think>...</think>
<schedule>
<event>
<name>...</name>
<start>...</start>
<end>...</end>
</event>
...
</schedule>

---

"""


def data_load():
    # Login using e.g. `huggingface-cli login` to access this dataset
    ds = load_dataset("wangzf/events-scheduling", split="train")
    logger.info(f"debug::ds[0]: {ds[0]}")

    ds = ds.map(
        lambda x: {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_PROMPT + x["prompt"]},
            ]
        }
    )
    logger.info(f"debug::ds[0]: {ds[0]}")

    return ds




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
