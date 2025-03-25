# -*- coding: utf-8 -*-

# ***************************************************
# * File        : gpt.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-01-25
# * Version     : 1.0.012519
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
import torch.nn as nn

from layers.transformer_block import TransformerBlock
from layers.layer_norm import LayerNorm
from utils.log_util import logger

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


class Model(nn.Module):

    def __init__(self, cfg):
        super(Model, self).__init__()

        # Embedding
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.emb_dim)
        self.pos_emb = nn.Embedding(cfg.context_length, cfg.emb_dim)
        self.drop_emb = nn.Dropout(cfg.dropout)
        # TransformerBlock
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg.n_layers)]
        )
        # LayerNorm
        self.final_norm = LayerNorm(cfg.emb_dim)
        # output head Linear
        self.out_head = nn.Linear(cfg.emb_dim, cfg.vocab_size, bias = False)

    def forward(self, in_idx):
        # in_idx size
        batch_size, seq_len = in_idx.shape
        # logger.info(f"batch_size: {batch_size} seq_len: {seq_len}")
        # embedding
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds  # shape: [batch_size, num_tokens, emb_size]
        x = self.drop_emb(x)  # dropout
        # transformer blocks
        x = self.trf_blocks(x)
        # final norm
        x = self.final_norm(x)
        # output head
        logits = self.out_head(x)

        return logits




# 测试代码 main 函数
def main():
    import tiktoken
    from utils.train_utils.gpt_generate import generate
    from utils.argsparser_tools import DotDict

    # model params
    GPT_CONFIG_124M = {
        "vocab_size": 50257,     # Vocabular size
        "context_length": 1024,  # Context length
        "max_new_toknes": 10,    # Maximum new tokens to generate
        "emb_dim": 768,          # Embedding dimension
        "n_heads": 12,           # Number of attention heads
        "n_layers": 12,          # Number of transformer layers
        "dropout": 0.1,          # Dropout rate
        "qkv_bias": False,       # Query-Key-Value bias
    }
    GPT_CONFIG_124M = DotDict(GPT_CONFIG_124M)

    # input data
    start_context = "Hello, I am"

    # tokenization
    tokenizer = tiktoken.get_encoding("gpt2")
    token_ids = tokenizer.encode(start_context)
    token_ids_tensor = torch.tensor(token_ids).unsqueeze(0)
    logger.info(f"\n{50 * '='}\n{22 * ' '}IN\n{50 * '='}")
    logger.info(f"Input text: {start_context}")
    logger.info(f"Encoded input text: {token_ids_tensor}")
    logger.info(f"Encoded input shape: {token_ids_tensor.shape}")

    # model
    torch.manual_seed(123)
    model = Model(GPT_CONFIG_124M)
    model.eval()  # disable dropout
    
    # generate text
    out = generate(
        model = model,
        token_idx = token_ids_tensor,
        max_new_tokens = GPT_CONFIG_124M["max_new_toknes"],
        context_size = GPT_CONFIG_124M["context_length"],
    )
    decoded_text = tokenizer.decode(out.squeeze(0).tolist())
    logger.info(f"\n{50 * '='}\n{22 * ' '}OUT\n{50 * '='}")
    logger.info(f"Output: {out}")
    logger.info(f"Outout shape: {out.shape}")
    logger.info(f"Output text: {decoded_text}")

if __name__ == "__main__":
    main()
