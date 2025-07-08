# -*- coding: utf-8 -*-

# ***************************************************
# * File        : hf_hub_login.py
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
from huggingface_hub import login


# 读取本地/项目的环境变量
_ = load_dotenv(find_dotenv())

# HuggingFace Hub token
HF_TOKEN_ID = os.environ["HF_TOKEN_ID"]

# Login
login(
    token=HF_TOKEN_ID,
    add_to_git_credential=True
)




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
