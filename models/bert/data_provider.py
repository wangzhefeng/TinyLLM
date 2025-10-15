# -*- coding: utf-8 -*-

# ***************************************************
# * File        : data_provider.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-10-15
# * Version     : 1.0.101523
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import os
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)
import warnings
warnings.filterwarnings("ignore")

from torch.utils.data import Dataset

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]
os.environ['LOG_NAME'] = LOGGING_LABEL
from utils.log_util import logger


class BDataset(Dataset):
    def __init__(self):
        pass

    def __getitem__(self, index):
        pass




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
