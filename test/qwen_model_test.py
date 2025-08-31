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

from utils.args_tools import DotDict
from utils.device import device_setting
from utils.model_memory import model_memory_size
from test.qwen_model_config import QWEN3_CONFIG
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
    device = device_setting()

    # config
    cfg = DotDict(QWEN3_CONFIG)

    # model
    model = Model(cfg=cfg)
    model.to(device)
    logger.info(f"model: \n{model}") 

    # model forward
    input_tensor = torch.tensor([1, 2, 3]).unsqueeze(0)
    input_tensor = input_tensor.to(device)
    logger.info(f"input_tensor: {input_tensor}")
    logger.info(f"input_tensor.shape: {input_tensor.shape}")

    output_tensor = model(input_tensor)
    logger.info(f"output_tensor: \n{output_tensor}")
    logger.info(f"output_tensor.shape: {output_tensor.shape}")

    model_memory_size(model, input_dtype=torch.float32, verbose=True)
    model_memory_size(model, input_dtype=torch.bfloat16, verbose=True)
    """
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total number of parameters: {total_params}")
    logger.info(f"Total number of parameters: {total_params / 1000**3:.1f}B")

    total_params_normalized = total_params - model.tok_embed.weight.numel()
    logger.info(f"Total number of unique parameters: {total_params_normalized}")
    logger.info(f"Total number of unique parameters: {total_params_normalized / 1000**3:.1f}B")

    # Calculate the total size in bytes (assuming float32, 4 bytes per parameter)
    total_size_bytes = total_params * 4
    # Convert to megabytes
    total_size_mb = total_size_bytes / (1024 * 1024)
    logger.info(f"Total size of the model: {total_size_mb:.2f}MB")
    """

if __name__ == "__main__":
    main()
