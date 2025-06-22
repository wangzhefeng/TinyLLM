# -*- coding: utf-8 -*-

# ***************************************************
# * File        : snippets.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-05-02
# * Version     : 1.0.050223
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# * TODO        : 1.
# ***************************************************

__all__ = []

# python libraries
import os
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)
from importlib.metadata import version

# global variable
LOGGING_LABEL = __file__.split('\\')[-1][:-3]
os.environ['LOG_NAME'] = LOGGING_LABEL

from utils.log_util import logger


# 1.package version
pkgs = [
    "matplotlib",
    "numpy",
    "tiktoken",
    "torch",
    "tensorflow",
    "pandas"
]
for p in pkgs:
    logger.info(f"{p} version: {version(p)}")


# 2.set options
import torch
torch.set_printoptions(sci_mode=False)






# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
