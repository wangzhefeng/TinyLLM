# -*- coding: utf-8 -*-

# ***************************************************
# * File        : test.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-06-19
# * Version     : 1.0.061921
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

__all__ = []

# python libraries
import os
import sys
ROOT = str(os.getcwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]
os.environ['LOG_NAME'] = LOGGING_LABEL

from utils.log_util import logger


try: 
    import torch
except: 
    raise ImportError('Install torch via `pip install torch`')

from packaging.version import Version as V
v = V(torch.__version__)
logger.info(f"torch version: {v}")

cuda = str(torch.version.cuda)
logger.info(f"cuda version: {cuda}")

is_ampere = torch.cuda.get_device_capability()[0] >= 8
logger.info(f"is ampere: {is_ampere}")

if cuda != "12.1" and cuda != "11.8" and cuda != "12.4": 
    raise RuntimeError(f"CUDA = {cuda} not supported!")
if   v <= V('2.1.0'): 
    raise RuntimeError(f"Torch = {v} too old!")
elif v <= V('2.1.1'): 
    x = 'cu{}{}-torch211'
elif v <= V('2.1.2'): 
    x = 'cu{}{}-torch212'
elif v  < V('2.3.0'): 
    x = 'cu{}{}-torch220'
elif v  < V('2.4.0'): 
    x = 'cu{}{}-torch230'
elif v  < V('2.5.0'): 
    x = 'cu{}{}-torch240'
elif v  < V('2.6.0'): 
    x = 'cu{}{}-torch250'
else: 
    raise RuntimeError(f"Torch = {v} too new!")

x = x.format(cuda.replace(".", ""), "-ampere" if is_ampere else "")
logger.info(f'pip install --upgrade pip && pip install "unsloth[{x}] @ git+https://github.com/unslothai/unsloth.git"')

"""
uv pip install --upgrade pip && pip install "unsloth[cu118-ampere-torch250] @ git+https://github.com/unslothai/unsloth.git"
"""




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
