# -*- coding: utf-8 -*-

# ***************************************************
# * File        : MLP.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-07-02
# * Version     : 1.0.070209
# * Description : MultilayerPerceptron
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

import torch.nn as nn

from utils.log_util import logger

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


class Model(nn.Module):
    
    def __init__(self, args):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(args.num_features, args.num_hidden_1),  # 784 * 128
            nn.ReLU(),
            nn.Linear(args.num_hidden_1, args.num_hidden_2),  # 128 * 256
            nn.ReLU(),
            nn.Linear(args.num_hidden_2, args.num_classes),  # 256 * 10
        )
 
    def forward(self, x):
        # x: (batch_size, 784)
        x = self.layers(x)
        # x: (batch_size, 10)

        return x






# 测试代码 main 函数
def main():
    # model arch params
    from utils.args_tools import DotDict
    args = DotDict({
        "num_features": 784,
        "num_hidden_1": 128,
        "num_hidden_2": 256,
        "num_classes": 10,
        "lora": False,
        "dora": False,
        "rank": 4,
        "alpha": 8,
    })
    # model without LoRA 
    model_pretrained = Model(args)
    logger.info(f"Model Arch without LoRA: \n{model_pretrained}")

if __name__ == "__main__":
    main()
