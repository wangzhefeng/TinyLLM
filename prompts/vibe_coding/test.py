# -*- coding: utf-8 -*-

# ***************************************************
# * File        : test.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-03-30
# * Version     : 1.0.033014
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# * TODO        : 1.
# ***************************************************

__all__ = []

# python libraries
import os
import sys
ROOT = str(os.getcwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)
import math
import random

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


def magic():
    if random.random() > 0.5:
        return "✨成功啦✨";
    else:
        raise ValueError("Oops")

print(magic())



# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
