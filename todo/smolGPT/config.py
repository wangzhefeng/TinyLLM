# -*- coding: utf-8 -*-

# ***************************************************
# * File        : config.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-01-31
# * Version     : 1.0.013122
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
from dataclasses import dataclass


import torch

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


@dataclass
class ModelConfig:
    block_size: int = 512
    vocab_size: int = 4096
    n_layer: int = 8
    n_head: int = 8
    n_embed: int = 512
    dropout: float = 0.2
    bias: bool = False


@dataclass
class TrainingConfig:
    learning_rate: float = 6e-4
    max_iters: int = 30000
    weight_decay: float = 1e-1
    beta1: float = 0.90
    beta2: float = 0.95
    grad_clip: float = 1.0
    
    decay_lr: bool = True
    warmup_iters: int = 1000
    lr_decay_iters: int = 30000
    min_lr: float = 6e-5
    
    eval_interval: int = 100
    log_interval: int = 10
    eval_iters: int = 200
    gradient_accumulation_steps: int = 4
    batch_size: int = 64
    
    device: str = str(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    dtype: str = "bfloat16"
    compile: bool = True




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
