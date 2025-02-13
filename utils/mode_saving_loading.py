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

from models.gpt import GPTModel
from utils.log_util import logger

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]
# device
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
logger.info(f"Using {device} device.")


# params
GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "dropout": 0.1,
    "qkv_bias": False,
}
# model
model = None
# optimizer
optimizer = None

# ------------------------------
# model weights
# ------------------------------
# model saving
torch.save(model.state_dict(), "model.pth")

# model loading
model = GPTModel(GPT_CONFIG_124M)
model.load_state_dict(torch.load("model.pth", map_location=device, weights_only=True))
model.eval();

# ------------------------------
# model weights and optimizer parameters
# ------------------------------
# model and optimizer weights saving
torch.save(
    {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    },
    "model_and_optimizer.pth",
)

# model and optimizer weights loading
checkpoint = torch.load("model_and_optimizer.pth", weights_only = True)
# model
model = GPTModel(GPT_CONFIG_124M)
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
