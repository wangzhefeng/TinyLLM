# -*- coding: utf-8 -*-

# ***************************************************
# * File        : data_to_markdown.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-05-02
# * Version     : 1.0.050220
# * Description : description
# * Link        : https://github.com/microsoft/markitdown
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


# global variable
LOGGING_LABEL = Path(__file__).name[:-3]
os.environ['LOG_NAME'] = LOGGING_LABEL

from markitdown import MarkItDown
from utils.log_util import logger


md = MarkItDown(enable_plugins=False)
result = md.convert("text.xlsx")
logger.info(f"result: {result}")







# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
