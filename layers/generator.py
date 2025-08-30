# -*- coding: utf-8 -*-

# ***************************************************
# * File        : generator.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2025-02-13
# * Version     : 0.1.021322
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

__all__ = [
    "generate_simple",
    "generate",
]

# python libraries
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)

import torch

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


def generate_simple(model, token_idx: torch.tensor, max_new_tokens: int, context_length: int):
    """
    generate text

    Args:
        model (_type_): LLM model
        token_idx (torch.tensor): token_idx is array of indices in the current contex. shape: (batch_ num_tokens)
        max_new_tokens (int): maximum length new tokens
        context_size (int): start context length
    """
    for i in range(max_new_tokens):
        # Crop current context if it exceeds the supported context size
        idx_cond = token_idx[:, -context_length:]
        # Get the predictions
        with torch.no_grad():
            logits = model(idx_cond)
        # Focus only on the last time step, shape:(batch_size,num_tokens,vocab_size)->(batch_size,vocab_size)
        logits = logits[:, -1, :]
        # Softmax, shape: (batch_size, vocab_size)
        logits = torch.softmax(logits, dim=-1)
        # Get the idx of the vocab entry with the highest probability value, shape: (batch_size, 1)
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        # Append sampled index to the running sequence, shape: (batch, n_tokens+1)
        token_idx = torch.cat((token_idx, idx_next), dim=1)

    return token_idx


def generate(model, token_idx: torch.tensor, max_new_tokens: int, context_length: int, 
             temperature: float=0.0, top_k: float=None, eos_id: int=None):
    """
    get logits, and only focus on last time step

    Args:
        model (_type_): LLM model
        token_idx (torch.tensor): token_idx is array of indices in the current contex. shape: (batch_size, num_tokens)
        max_new_tokens (int): maximum length new tokens
        context_size (int): start context length
    """
    for i in range(max_new_tokens):
        # Crop current context if it exceeds the supported context size
        idx_cond = token_idx[:, -context_length:]
        # Get the predictions
        with torch.no_grad():
            logits = model(idx_cond)
        # Focus only on the last time step, shape:(batch_size,num_tokens,vocab_size)->(batch_size,vocab_size)
        logits = logits[:, -1, :]
        # Filter logits with top_k sampling
        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(logits < min_val, torch.tensor(float("-inf")).to(logits.device), logits)
        # Apply temperature scaling
        if temperature > 0.0:
            logits = logits / temperature
            # apply softmax to get probabilities
            probs = torch.softmax(logits, dim=-1)  # shape:(batch_size, context_length)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # shape:(batch_size, 1)
        # Otherwise same as before: get idx of the vocab entry with the highest logits value
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # shape:(batch_size, 1)
        # Stop generating early if end-of-sequence token is encountered and eos_id is specified
        if idx_next == eos_id:
            break
        # Append sampled index to the running sequence
        token_idx = torch.cat([token_idx, idx_next], dim=1)  # (batch_size, num_tokens+1)

    return token_idx




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
