# -*- coding: utf-8 -*-

# ***************************************************
# * File        : tools.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-01-13
# * Version     : 1.0.011322
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# * TODO        : 1.
# ***************************************************

__all__ = []

# python libraries
import os
import sys
ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


class DotDict(dict):

    """
    dot.notation access to dictionary attributes
    """
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, dct):
        for key, value in dct.items():
            if hasattr(value, 'keys'):
                value = DotDict(value)
            self[key] = value




# 测试代码 main 函数
def main():
    dct = {
        'scalar_value': 1, 
        'nested_dict': {
            'value': 2, 
            'nested_nested': {
                'x': 21
            }
        }
    }
    dct = DotDict(dct)

    print(dct.nested_dict.nested_nested.x)

if __name__ == "__main__":
    main()
