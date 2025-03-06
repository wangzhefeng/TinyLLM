# -*- coding: utf-8 -*-

# ***************************************************
# * File        : model_finetune_classification.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2025-03-04
# * Version     : 0.1.030423
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

import torch.nn as nn

from layers.lora import replace_linear_with_lora
from utils.log_util import logger

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


def finetune_model_simple(model, device, emb_dim: int, num_classes: int):
    """
    add a classification head
    """
    # build model and print model architecture
    logger.info(f"model: \n{model}")  # model architecture
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total trainable parameters before freeze: {total_params}")

    # freeze model(make all layers non-trainable)
    for param in model.parameters():
        param.requires_grad = False
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total trainable parameters after freeze: {total_params}")

    # replace output layer
    model.out_head = nn.Linear(
        in_features = emb_dim, 
        out_features = num_classes
    )

    # make the last transformer block and final LayerNorm module 
    # connecting the last transformer block to the output layer trainable
    for param in model.trf_blocks[-1].parameters():
        param.requires_grad = True

    for param in model.final_norm.parameters():
        param.requires_grad = True

    # move model to device 
    model.to(device) 
    logger.info(f"model: \n{model}")  # model architecture

    return model


def finetune_model_lora(model, device, emb_dim: int, num_classes: int):
    """
    add a classification head
    """
    # build model and print model architecture
    logger.info(f"model: \n{model}")  # model architecture
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total trainable parameters before freeze: {total_params}")

    # freeze model(make all layers non-trainable)
    for param in model.parameters():
        param.requires_grad = False
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total trainable parameters after freeze: {total_params}")

    # replace output layer
    model.out_head = nn.Linear(
        in_features = emb_dim, 
        out_features = num_classes
    )

    # TODO replace linear with LinearWithLoRA
    replace_linear_with_lora(model, rank = 16, alpha = 16)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total trainable LoRA parameters: {total_params}")

    # make the last transformer block and final LayerNorm module 
    # connecting the last transformer block to the output layer trainable
    for param in model.trf_blocks[-1].parameters():
        param.requires_grad = True

    for param in model.final_norm.parameters():
        param.requires_grad = True

    # move model to device 
    model.to(device) 
    logger.info(f"model: \n{model}")  # model architecture

    return model




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
