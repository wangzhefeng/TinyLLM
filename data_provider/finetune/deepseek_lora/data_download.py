# -*- coding: utf-8 -*-

# ***************************************************
# * File        : data_download.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-03-01
# * Version     : 0.1.030115
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

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from modelscope.msdatasets import MsDataset

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


# data download
ds = MsDataset.load(
    "AI-ModelScope/medical-o1-reasoning-SFT", 
    trust_remote_code = True
)




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
