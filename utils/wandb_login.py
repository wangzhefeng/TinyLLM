# -*- coding: utf-8 -*-

# ***************************************************
# * File        : wandb_login.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-07-08
# * Version     : 1.0.070817
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import os
from dotenv import find_dotenv, load_dotenv
import wandb


# 读取本地/项目的环境变量
_ = load_dotenv(find_dotenv())

# HuggingFace Hub token
WANDB_TOKEN_ID = os.environ["WANDB_TOKEN_ID"]

# Login
wandb.login(
    key=WANDB_TOKEN_ID,
    # relogin=True,
)




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
