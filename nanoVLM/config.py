# -*- coding: utf-8 -*-

# ***************************************************
# * File        : config.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-07-08
# * Version     : 1.0.070816
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

__all__ = []

# python libraries
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)
from dataclasses import dataclass
import warnings
warnings.filterwarnings("ignore")

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


@dataclass
class VLMConfig:
    pass


@dataclass
class TrainConfig:
    pass




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
