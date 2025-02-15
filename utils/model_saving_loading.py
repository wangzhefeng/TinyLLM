# -*- coding: utf-8 -*-

# ***************************************************
# * File        : mode_saving_loading.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-01-29
# * Version     : 1.0.012906
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# * TODO        : 1.
# ***************************************************

__all__ = []

# python libraries
import os
import sys
ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import torch

from models.gpt import Model

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


def save_model_weights(model, model_path: str):
    """
    Save model weights
    """
    # model saving
    torch.save(model.state_dict(), model_path)


def load_model_weights(args, model_path: str, device: str):
    """
    Load model weights
    """
    model = Model(args)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval();


def save_model_optim_weights(model, optimizer, model_path: str):
    """
    Save model weights and optimizer parameters
    """
    # model and optimizer weights saving
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        model_path,
    )


def load_model_optim_weights(args, model_path: str):
    """
    Load model weights and optimizer parameters
    """
    # model and optimizer weights loading
    checkpoint = torch.load(model_path, weights_only = True)
    # model
    model = Model(args)
    model.load_state_dict(checkpoint["model_state_dict"])
    # optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr = 0.0005, weight_decay = 0.1)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    model.train();




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
