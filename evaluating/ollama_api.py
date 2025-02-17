# -*- coding: utf-8 -*-

# ***************************************************
# * File        : ollama_api.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2025-02-17
# * Version     : 0.1.021723
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
import psutil

from utils.log_util import logger

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]



def check_if_running(process_name):
    running = False
    for proc in psutil.process_iter(["name"]):
        if process_name in proc.info["name"]:
            running = True
            break
    return running


# check ollama running
ollama_running = check_if_running("ollama")
logger.info(f"Ollama running: {ollama_running}")

if not ollama_running:
    raise RuntimeError("Ollama not running. Launch ollama before proceeding.")




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
