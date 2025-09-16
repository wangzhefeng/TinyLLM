# -*- coding: utf-8 -*-

# ***************************************************
# * File        : decoding.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2025-02-13
# * Version     : 0.1.021322
# * Description : Decoding Strategies
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

__all__ = [
    "generate_simple",
    "generate_simple_cached",
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
    for _ in range(max_new_tokens):
        # Crop current context if it exceeds the supported context size
        cond_idx = token_idx[:, -context_length:]
        # Get the predictions
        with torch.no_grad():
            logits = model(cond_idx)
        # Focus only on the last time step, shape:(batch_size,num_tokens,vocab_size)->(batch_size,vocab_size)
        logits = logits[:, -1, :]
        # Softmax, shape: (batch_size, vocab_size)
        logits = torch.softmax(logits, dim=-1)
        # Get the idx of the vocab entry with the highest probability value, shape: (batch_size, 1)
        next_idx = torch.argmax(logits, dim=-1, keepdim=True)
        # Append sampled index to the running sequence, shape: (batch, n_tokens+1)
        token_idx = torch.cat((token_idx, next_idx), dim=1)

    return token_idx


def generate_simple_cached(model, token_idx: torch.tensor, max_new_tokens: int, context_length: int, use_cache=True):
    """
    generate text

    Args:
        model (_type_): LLM model
        token_idx (torch.tensor): token_idx is array of indices in the current contex. shape: (batch_ num_tokens)
        max_new_tokens (int): maximum length new tokens
        context_size (int): start context length
    """
    model.eval()
    ctx_len = context_length or model.pos_embed.num_embeddings
    with torch.no_grad():
        if use_cache:
            # Init cache with full prompt
            model.reset_kv_cache()
            # Crop current context if it exceeds the supported context size
            cond_idx = token_idx[:, -ctx_len:]
            # Get the predictions
            logits = model(cond_idx, use_cache)
            for _ in range(max_new_tokens):
                # a) pick the token with the highest log-probability(greedy sampling)
                # -------------------
                # Focus only on the last time step, shape:(batch_size,num_tokens,vocab_size)->(batch_size,vocab_size)
                # logits = logits[:, -1, :]
                logits = logits[:, -1]
                # Softmax, shape: (batch_size, vocab_size)
                # logits = torch.softmax(logits, dim=-1)
                # Get the idx of the vocab entry with the highest probability value, shape: (batch_size, 1)
                next_idx = torch.argmax(logits, dim=-1, keepdim=True)
                # b) append it to the running sequence
                # -------------------
                # Append sampled index to the running sequence, shape: (batch, n_tokens+1)
                token_idx = torch.cat((token_idx, next_idx), dim=1)
                # c) feed model only the new token
                # -------------------
                logits = model(next_idx, use_cache)
        else:
            for _ in range(max_new_tokens):
                # Crop current context if it exceeds the supported context size
                cond_idx = token_idx[:, -ctx_len:]
                # Get the predictions
                logits = model(cond_idx, use_cache)
                # Focus only on the last time step, shape:(batch_size,num_tokens,vocab_size)->(batch_size,vocab_size)
                # logits = logits[:, -1, :]
                logits = logits[:, -1]
                # Softmax, shape: (batch_size, vocab_size)
                # logits = torch.softmax(logits, dim=-1)
                # Get the idx of the vocab entry with the highest probability value, shape: (batch_size, 1)
                next_idx = torch.argmax(logits, dim=-1, keepdim=True)
                # Append sampled index to the running sequence, shape: (batch, n_tokens+1)
                token_idx = torch.cat((token_idx, next_idx), dim=1)

    return token_idx


def generate(model, token_idx: torch.tensor, max_new_tokens: int, context_length: int, 
             temperature: float=0.0, top_k: int=None, eos_id: int=None, use_cache=False):
    """
    get logits, and only focus on last time step

    Args:
        model (_type_): LLM model
        token_idx (torch.tensor): token_idx is array of indices in the current contex. shape: (batch_size, num_tokens)
        max_new_tokens (int): maximum length new tokens
        context_size (int): start context length
    """
    model.eval()
    ctx_len = context_length or model.pos_embed.num_embeddings
    with torch.no_grad():
        if use_cache:
            # Init cache with full prompt
            model.reset_kv_cache()
            # Crop current context if it exceeds the supported context size
            cond_idx = token_idx[:, -ctx_len:]
            # Get the predictions
            logits = model(cond_idx, use_cache)
            for _ in range(max_new_tokens):
                # Focus only on the last time step, shape:(batch_size,num_tokens,vocab_size)->(batch_size,vocab_size)
                # logits = logits[:, -1, :]
                logits = logits[:, -1]
                # Filter logits with top_k sampling
                if top_k is not None:
                    top_logits, _ = torch.topk(logits, top_k)
                    min_val = top_logits[:, -1]
                    logits = torch.where(logits < min_val, torch.tensor(float("-inf")).to(logits.device), logits)
                # Apply temperature scaling
                if temperature > 0.0:
                    logits = logits / temperature
                    # apply softmax to get probabilities
                    # probs = torch.softmax(logits, dim=-1)                # shape:(batch_size, context_length)
                    # sample from the distribution
                    next_idx = torch.multinomial(logits, num_samples=1)    # shape:(batch_size, 1)
                # Otherwise same as before: get idx of the vocab entry with the highest logits value
                else:
                    next_idx = torch.argmax(logits, dim=-1, keepdim=True)  # shape:(batch_size, 1)
                # Stop generating early if end-of-sequence token is encountered and eos_id is specified
                if next_idx == eos_id:
                    break
                # Append sampled index to the running sequence
                token_idx = torch.cat([token_idx, next_idx], dim=1)        # shape: (batch_size, num_tokens+1)
                # Feed model only the new token
                logits = model(next_idx, use_cache)
        else:
            for _ in range(max_new_tokens):
                # Crop current context if it exceeds the supported context size
                cond_idx = token_idx[:, -ctx_len:]
                # Get the predictions
                logits = model(cond_idx, use_cache=False)
                # Focus only on the last time step, shape:(batch_size,num_tokens,vocab_size)->(batch_size,vocab_size)
                # logits = logits[:, -1, :]
                logits = logits[:, -1]
                # Filter logits with top_k sampling
                if top_k is not None:
                    top_logits, _ = torch.topk(logits, top_k)
                    min_val = top_logits[:, -1]
                    logits = torch.where(logits < min_val, torch.tensor(float("-inf")).to(logits.device), logits)
                # Apply temperature scaling
                if temperature > 0.0:
                    logits = logits / temperature
                    # apply softmax to get probabilities
                    # probs = torch.softmax(logits, dim=-1)                # shape:(batch_size, context_length)
                    # sample from the distribution
                    next_idx = torch.multinomial(logits, num_samples=1)    # shape:(batch_size, 1)
                # Otherwise same as before: get idx of the vocab entry with the highest logits value
                else:
                    next_idx = torch.argmax(logits, dim=-1, keepdim=True)  # shape:(batch_size, 1)
                # Stop generating early if end-of-sequence token is encountered and eos_id is specified
                if eos_id is not None and torch.all(next_idx == eos_id):
                    break
                
                yield next_idx

                # Append sampled index to the running sequence
                token_idx = torch.cat([token_idx, next_idx], dim=1)        # shape: (batch_size, num_tokens+1)

    return token_idx




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
