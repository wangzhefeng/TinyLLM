# -*- coding: utf-8 -*-

# ***************************************************
# * File        : layernorm_test.py
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

from layers.normailzation.layer_norm import LayerNorm, LayerNormPyTorch

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]
os.environ['LOG_NAME'] = LOGGING_LABEL
from utils.log_util import logger




# 测试代码 main 函数
def main():
    torch.set_printoptions(sci_mode=False)
    # ------------------------------
    # Layer Norm test
    # ------------------------------
    # data
    torch.manual_seed(123)
    batch_example = torch.randn(2, 5)  # create 2 training examples with 5 dimensions(features) each
    logger.info(f"batch_example: \n{batch_example}")
    
    # layer norm
    ln = LayerNorm(embed_dim=5)
    out_ln = ln(batch_example)
    logger.info(f"out_ln: \n{out_ln}")
    
    mean = out_ln.mean(dim=-1, keepdim=True)
    var = out_ln.var(dim=-1, unbiased=False, keepdim=True)
    logger.info(f"Mean: \n{mean}")
    logger.info(f"Variance: \n{var}")
    # ------------------------------
    # PyTorch Layer Norm test
    # ------------------------------
    # pytorch layer norm
    ln = LayerNormPyTorch(embed_dim=5)
    out_ln = ln(batch_example)
    logger.info(f"out_ln: \n{out_ln}")
    
    mean = out_ln.mean(dim=-1, keepdim=True)
    var = out_ln.var(dim=-1, unbiased=False, keepdim=True)
    logger.info(f"Mean: \n{mean}")
    logger.info(f"Variance: \n{var}")

if __name__ == "__main__":
    main()
