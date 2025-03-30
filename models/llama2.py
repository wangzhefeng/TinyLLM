# -*- coding: utf-8 -*-

# ***************************************************
# * File        : llama.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2025-03-04
# * Version     : 0.1.030400
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

from layers.transformer_block import TransformerBlockLlama
from layers.rms_norm import RMSNorm

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


class Model(nn.Module):

    def __init__(self, cfg):
        super(Model, self).__init__()

        # embedding
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.emb_dim, dtype=cfg.dtype)
        # transformer block
        self.trf_blocks = nn.Sequential(
            *[TransformerBlockLlama(cfg) for _ in range(cfg.n_layers)]
        )
        # RMSNorm
        self.final_norm = RMSNorm(cfg.emb_dim)
        # output head linear
        self.out_head = nn.Linear(cfg.emb_dim, cfg.vocab_size, bias = False, dtype=cfg.dtype)
    
    def forward(self, in_idx):
        # embedding
        tok_embeds = self.tok_emb(in_idx)
        x = tok_embeds
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)

        return logits




# 测试代码 main 函数
def main():
    import torch
    from models.llama2 import Model
    from utils.args_tools import DotDict
    from utils.train_utils.gpt_generate import generate
    from tokenizer.tokenization import (
        text_to_token_ids,
        token_ids_to_text,
    )
    from utils.device import device_setting
    from utils.log_util import logger

    # model params
    LLAMA2_CONFIG_7B = {
        "vocab_size": 32000,     # Vocabulary size
        "context_length": 4096,  # Context length
        "emb_dim": 4096,         # Embedding dimension
        "n_heads": 32,           # Number of attention heads
        "n_layers": 32,          # Number of layers
        "hidden_dim": 11008,     # NEW: Size of the intermediate dimension in FeedForward
        "dtype": torch.bfloat16  # NEW: Lower-precision dtype to reduce memory usage
    }
    LLAMA2_CONFIG_7B = DotDict(LLAMA2_CONFIG_7B)
    # model
    model = Model(LLAMA2_CONFIG_7B)
    # device
    device = device_setting()
    # input text
    input_text = "Every effort moves"

    # model generate
    token_ids = generate(
        model = model.to(device),
        token_idx = text_to_token_ids(input_text).to(device),
        max_new_tokens = 30,
        context_size = LLAMA2_CONFIG_7B.context_length,
        eos_id = 50256,
    )
    generated_text = token_ids_to_text(token_ids)
    logger.info(f"generated_text:\n{generated_text}")

if __name__ == "__main__":
    main()
