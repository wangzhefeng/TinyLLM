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


class GPT(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        # Embedding
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["dropout"])
        # TransformerBlock
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        # LayerNorm
        self.final_norm = LayerNorm(cfg["emb_dim"])
        # output head Linear
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias = False)

    def forward(self, in_idx):
        # in_idx size
        batch_size, seq_len = in_idx.shape
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


def generate_text_simple(model, 
                         idx: torch.tensor, 
                         max_new_tokens: int, 
                         context_size: int):
    """
    generate text

    Args:
        model (_type_): LLM model
        idx (torch.tensor): idx is array of indices in the current contex. shape: (batch, n_tokens)
        max_new_tokens (int): maximum length new tokens
        context_size (int): start context length

    Returns:
        _type_: _description_
    """
    for _ in range(max_new_tokens):
        # crop current context if it exceeds the supported context size
        logger.info(f"idx before crop: {idx}")
        idx_cond = idx[:, -context_size:]
        logger.info(f"idx after crop: {idx_cond}")
        # get the predictions
        with torch.no_grad():
            logits = model(idx_cond)
        logger.info(f"logits: {logits}")
        # focus only on the last time step
        # shape: (batch, n_tokens, vocab_size) -> (batch, vocab_size)
        logits = logits[:, -1, :]
        logger.info(f"logits: {logits}")
        # softmax
        probas = torch.softmax(logits, dim=-1)  # (batch, vocab_size)
        logger.info(f"probas: {probas}")
        # get the idx of the vocab entry with the highest probability value
        idx_next = torch.argmax(probas, dim = -1, keepdim = True)
        logger.info(f"idx_next: {idx_next}")
        # append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim = 1)  # (batch, n_tokens+1)
        logger.info(f"idx: {idx}\n")

    return idx


def generate(model, 
             idx: torch.tensor, 
             max_new_tokens: int, 
             context_size: int, 
             temperature: float=0.0, 
             top_k: float=None, 
             eos_id: int=None):
    """
    get logits, and only focus on last time step

    Args:
        model (_type_): _description_
        idx (torch.tensor): _description_
        max_new_tokens (int): _description_
        context_size (int): _description_
        temperature (float, optional): _description_. Defaults to 0.0.
        top_k (float, optional): _description_. Defaults to None.
        eos_id (int, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    for _ in range(max_new_tokens):
        # crop current context if it exceeds the supported context size
        logger.info(f"idx before crop: {idx}")
        idx_cond = idx[:, -context_size:]
        logger.info(f"idx after crop: {idx_cond}")
        # get the predictions
        with torch.no_grad():
            logits = model(idx_cond)
        logger.info(f"logits: {logits}")
        # focus only on the last time step
        # shape: (batch, n_tokens, vocab_size) -> (batch, vocab_size)
        logits = logits[:, -1, :]
        # filter logits with top_k sampling
        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(
                logits < min_val, 
                torch.tensor(float("-inf")).to(logits.device),
                logits
            )
        # apply temperature scaling
        if temperature > 0.0:
            logits = logits / temperature
            # apply softmax to get probabilities
            probs = torch.softmax(logits, dim=-1)  # (batch_size, context_len)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)
        # otherwise same as before: get idx of the vocab entry with the highest logits value
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch_size, 1)
        # stop generating early if end-of-sequence token is encountered and eos_id is specified
        if idx_next == eos_id:
            break
        # append sampled index to the running sequence
        idx = torch.cat([idx, idx_next], dim=1)  # (batch_size, num_tokens+1)

    return idx




# 测试代码 main 函数
def main():
    import tiktoken

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
    model = GPT(GPT_CONFIG_124M)
    model.eval()  # disable dropout
    
    # generate text
    out = generate_text_simple(
        model = model,
        idx = token_ids_tensor,
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
