# -*- coding: utf-8 -*-

# ***************************************************
# * File        : feed_forward_test.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-08-30
# * Version     : 1.0.083016
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

from layers.feed_forward import (
    FeedForwardReLU,
    FeedForwardGELU, 
    FeedForwardSiLU,
)

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]
os.environ['LOG_NAME'] = LOGGING_LABEL
from utils.log_util import logger




# 测试代码 main 函数
def main():
    from test.model_config import device, tokenizer, GPT2_124M_CONFIG

    # feed forward layer
    ffn_relu = FeedForwardReLU(GPT2_124M_CONFIG)
    ffn_gelu = FeedForwardGELU(GPT2_124M_CONFIG)
    logger.info(f"ffn_gelu: \n{ffn_gelu}")
    ffn_silu = FeedForwardSiLU(GPT2_124M_CONFIG)
    
    # input tensor
    x = torch.rand(2, 3, 768)
    logger.info(f"x: \n{x}")
    logger.info(f"x.shape: {x.shape}") 
    
    # forward
    out_relu = ffn_relu(x)
    out_gelu = ffn_gelu(x)
    out_silu = ffn_silu(x)
    logger.info(f"out_relu: \n{out_relu}")
    logger.info(f"out_gelu: \n{out_gelu}")
    logger.info(f"out_silu: \n{out_silu}")
    logger.info(f"out_relu.shape: {out_relu.shape}")
    logger.info(f"out_gelu.shape: {out_gelu.shape}")
    logger.info(f"out_silu.shape: {out_silu.shape}")

if __name__ == "__main__":
    main()
