# -*- coding: utf-8 -*-

# ***************************************************
# * File        : qwen_model_test.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-08-31
# * Version     : 1.0.083123
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

__all__ = []

# python libraries
import os
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)
import warnings
warnings.filterwarnings("ignore")

import torch

from utils.device import device_setting
from utils.model_memory import model_memory_size
from config.qwen3_model_cfg.model_cfgs import QWEN3_CONFIG
from models.qwen3_06B import Model

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]
os.environ['LOG_NAME'] = LOGGING_LABEL
from utils.log_util import logger




# 测试代码 main 函数
def main():
    # random seed
    torch.manual_seed(123)

    # device
    device = device_setting(verbose=True)

    # model
    model = Model(cfg=QWEN3_CONFIG)
    model.to(device)
    logger.info(f"model: \n{model}") 

    # model forward
    input_tensor = torch.tensor([1, 2, 3]).unsqueeze(0)
    input_tensor = input_tensor.to(device)
    logger.info(f"input_tensor: {input_tensor}")
    logger.info(f"input_tensor shape: {input_tensor.shape}")

    output_tensor = model(input_tensor)
    logger.info(f"output_tensor: \n{output_tensor}")
    logger.info(f"output_tensor shape: {output_tensor.shape}")

    model_memory_size(model, input_dtype=torch.float32, verbose=True)
    model_memory_size(model, input_dtype=torch.bfloat16, verbose=True)

if __name__ == "__main__":
    main()
