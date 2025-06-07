# -*- coding: utf-8 -*-

# ***************************************************
# * File        : prompt_demo.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2025-02-23
# * Version     : 0.1.022321
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import os
import sys
ROOT = str(os.getcwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)

from peft import (
    get_peft_config, 
    get_peft_model, 
    PromptTuningInit, 
    PromptTuningConfig,
    TaskType,
    PeftType,
)

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


# 




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
