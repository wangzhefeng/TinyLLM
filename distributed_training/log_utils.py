# -*- coding: utf-8 -*-

# ***************************************************
# * File        : log_utils.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-07-08
# * Version     : 1.0.070814
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

__all__ = []

# python libraries
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)
import logging
import warnings
warnings.filterwarnings("ignore")

import torch

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


logging.basicConfig(
    format="%(asctime)s %(message)s", 
    datefmt="%m/%d/%Y %I:%M:%S %p", 
    level=logging.INFO
)


def get_logger():
    return logging.getLogger(__name__)


def rank_log(_rank, logger, msg):
    """
    helper function to long only on global rank 0

    Args:
        _rank (_type_): _description_
        logger (_type_): _description_
        msg (_type_): _description_
    """
    if _rank == 0:
        logger.info(f" {msg}")


def verify_min_gpu_count(min_gpus: int=2) -> bool:
    """
    verification that we have at least 2 gpus to run dist examples

    Args:
        min_gpus (int, optional): _description_. Defaults to 2.

    Returns:
        bool: _description_
    """
    has_cuda = torch.cuda.is_available()
    gpu_count = torch.cuda.device_count()

    return has_cuda and gpu_count >= min_gpus




# 测试代码 main 函数
def main():
    res = get_logger()

if __name__ == "__main__":
    main()
