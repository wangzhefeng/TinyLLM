# -*- coding: utf-8 -*-

# ***************************************************
# * File        : model_finetune_classification.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-03-04
# * Version     : 0.1.030423
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import os
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)

import torch.nn as nn

from layers.lora_dora import replace_linear_with_lora
from utils.log_util import logger

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


def finetune_model(model, emb_dim: int, num_classes: int, finetune_method: str):
    """
    add a classification head
    """
    # model architecture before finetune
    logger.info(f"model architecture before finetune: \n{model}") 

    # model params numbers before freeze
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total trainable parameters before freeze: {total_params}")
    # freeze model(make all layers non-trainable)
    for param in model.parameters():
        param.requires_grad = False
    # model params numbers after freeze
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total trainable parameters after freeze: {total_params}")

    # replace output layer
    model.out_head = nn.Linear(
        in_features = emb_dim, 
        out_features = num_classes
    )
    
    # replace linear with LinearWithLoRA
    if finetune_method == "lora":
        replace_linear_with_lora(model, rank = 16, alpha = 16)
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Total trainable LoRA parameters: {total_params}")

    # make the last transformer block and final LayerNorm module 
    # connecting the last transformer block to the output layer trainable
    for param in model.trf_blocks[-1].parameters():
        param.requires_grad = True

    for param in model.final_norm.parameters():
        param.requires_grad = True

    # model architecture after finetune
    logger.info(f"model architecture after finetune: \n{model}")

    return model




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
