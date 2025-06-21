# -*- coding: utf-8 -*-

# ***************************************************
# * File        : data_download.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-05-03
# * Version     : 1.0.050300
# * Description : description
# * Link        : https://www.modelscope.cn/datasets/gongjy/minimind_dataset/files
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
from pathlib import Path

from modelscope.msdatasets import MsDataset

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


ds =  MsDataset.load(
    'gongjy/minimind_dataset', 
    subset_name='default', 
    split='train'
)



# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
