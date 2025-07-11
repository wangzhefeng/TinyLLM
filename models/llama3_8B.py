# -*- coding: utf-8 -*-

# ***************************************************
# * File        : llama3.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-03-21
# * Version     : 1.0.032122
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)

import torch
import torch.nn as nn

from layers.transformer_block import TransformerBlockLlama3
from layers.rms_norm import RMSNorm

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


class Model(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        # embedding
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.emb_dim, dtype=cfg.dtype)
        # transformer block
        self.trf_blocks = nn.Sequential(
            *[TransformerBlockLlama3(cfg) for _ in range(cfg.n_layers)]
        )
        # RMSNorm
        self.final_norm = RMSNorm(cfg.emb_dim)
        # output head linear
        self.out_head = nn.Linear(cfg.emb_dim, cfg.vocab_size, bias = False, dtype=cfg.dtype)
    
    def forward(self, in_idx):
        # TODO in_idx size
        batch_size, seq_len = in_idx.shape
        # embedding
        tok_embeds = self.tok_emb(in_idx)
        x = tok_embeds
        # transformer blocks
        x = self.trf_blocks(x)
        # final rms norm
        x = self.final_norm(x)
        # TODO output head
        logits = self.out_head(x.to(torch.bfloat16))

        return logits




# 测试代码 main 函数
def main():
    import torch
    from utils.args_tools import DotDict
    from utils.llm.gpt_generate import generate
    from tokenizers.tokenization import (
        text_to_token_ids,
        token_ids_to_text,
    )
    from model_load.model_cfgs import LLAMA3_CONFIG_8B
    from utils.model_memory import model_memory_size
    from utils.device import device_setting
    from utils.log_util import logger

    # model params 
    LLAMA3_CONFIG_8B = DotDict(LLAMA3_CONFIG_8B)

    # model
    model = Model(LLAMA3_CONFIG_8B)

    # device
    device = device_setting()
    # ------------------------------
    # model memory size
    # ------------------------------
    total_memory_gb = model_memory_size(model, input_dtype=torch.float32)
    total_memory_gb = model_memory_size(model, input_dtype=torch.bfloat16)
    # ------------------------------
    # inference
    # ------------------------------
    # input text
    input_text = "Every effort moves"

    # model generate
    token_ids = generate(
        model = model.to(device),
        token_idx = text_to_token_ids(input_text, tokenizer_model="llama3-8b").to(device),
        max_new_tokens = 30,
        context_size = LLAMA3_CONFIG_8B.context_length,
        eos_id = 50256,  # TODO
    )
    generated_text = token_ids_to_text(token_ids, tokenizer_model="llama3-8b")
    logger.info(f"generated_text:\n{generated_text}")

if __name__ == "__main__":
    main()
