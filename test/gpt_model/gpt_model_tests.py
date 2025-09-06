# -*- coding: utf-8 -*-

# ***************************************************
# * File        : tests.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-02-13
# * Version     : 0.1.021300
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

import torch

from layers.transformer_block import TransformerBlockGPT2_124M
from models.gpt2_124M import Model

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]
os.environ['LOG_NAME'] = LOGGING_LABEL
from utils.log_util import logger


def gpt2_124M_transformer_block_test(GPT2_124M_CONFIG):
    # input
    torch.manual_seed(123)
    x = torch.rand(2, 4, 768)

    # transformer
    block = TransformerBlockGPT2_124M(GPT2_124M_CONFIG)
    output = block(x)
    logger.info(f"Input shape: {x.shape}")
    logger.info(f"Output shape: {output.shape}")


def gpt2_model_test(tokenizer, GPT2_CONFIG, device):  
    # ------------------------------
    # input text
    # ------------------------------
    txt1 = "Every effort moves you"
    txt2 = "Every day holds a"
    # ------------------------------
    # input batch
    # ------------------------------
    batch = []
    batch.append(torch.tensor(tokenizer.encode(txt1)))
    batch.append(torch.tensor(tokenizer.encode(txt2)))
    logger.info(f"batch: \n{batch}")
    batch = torch.stack(batch, dim=0)
    batch = batch.to(device)
    logger.info(f"batch: \n{batch}")
    logger.info(f"batch.shape: {batch.shape}")
    # ------------------------------
    # model
    # ------------------------------
    model = Model(GPT2_CONFIG).to(device)
    # ------------------------------
    # model size and memory
    # ------------------------------
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"total_params: {total_params / 1000**3:.1f}B")
    total_params_without_outhead = total_params - sum(p.numel() for p in model.out_head.parameters())
    logger.info(f"total_params_without_outhead: {total_params_without_outhead / 1000**3:.1f}B")
    # Calculate the total size in bytes (assuming float32, 4 bytes per parameter)
    total_size_bytes = total_params * 4
    # Convert to megabytes
    total_size_mb = total_size_bytes / (1024 * 1024)
    logger.info(f"Total size of the model: {total_size_mb:.2f}MB")
    # ------------------------------
    # model inference
    # ------------------------------
    # disable dropout during inference
    model.eval()
    # model inference
    logits = model(batch)
    logger.info(f"logits: \n{logits}")
    logger.info(f"logits.shape: {logits.shape}")




# 测试代码 main 函数
def main():
    from test.gpt_model.model_config import (
        device, 
        tokenizer, 
        GPT2_124M_CONFIG,
        GPT2_MEDIUM_CONFIG,
        GPT2_LARGE_CONFIG,
        GPT2_XL_CONFIG,
    )

    # GPT2 small
    gpt2_124M_transformer_block_test(GPT2_124M_CONFIG)
    gpt2_model_test(tokenizer, GPT2_124M_CONFIG, device)
    # GPT2 medium
    gpt2_model_test(tokenizer, GPT2_MEDIUM_CONFIG, device)
    # GPT2 large
    gpt2_model_test(tokenizer, GPT2_LARGE_CONFIG, device)
    # GPT2 XL
    gpt2_model_test(tokenizer, GPT2_XL_CONFIG, device)

if __name__ == "__main__":
    main()
