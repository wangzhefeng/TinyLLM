# -*- coding: utf-8 -*-

# ***************************************************
# * File        : main.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-07-06
# * Version     : 1.0.070602
# * Description : description
# * Link        : https://github.com/rasbt/LLM-finetuning-scripts/blob/main/conventional/distilbert-movie-review/3_finetuning-all-layers.ipynb
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
import warnings
warnings.filterwarnings("ignore")

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]
os.environ['LOG_NAME'] = LOGGING_LABEL
from utils.log_util import logger







# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
