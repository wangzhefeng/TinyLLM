# -*- coding: utf-8 -*-

# ***************************************************
# * File        : data_download.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2025-03-01
# * Version     : 0.1.030115
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import os
import sys
ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from modelscope.msdatasets import MsDataset

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


# data download
ds = MsDataset.load('AI-ModelScope/medical-o1-reasoning-SFT', trust_remote_code = True)






# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
