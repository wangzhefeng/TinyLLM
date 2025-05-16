# -*- coding: utf-8 -*-

# ***************************************************
# * File        : gpt_generate.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2025-02-13
# * Version     : 0.1.021322
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

import torch

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


def generate_simple_V2(
        model, 
        token_idx: torch.tensor, 
        max_new_tokens: int, 
        context_size: int
    ):
    """
    generate text

    Args:
        model (_type_): LLM model
        token_idx (torch.tensor): token_idx is array of indices in the current contex. shape: (batch, n_tokens)
        max_new_tokens (int): maximum length new tokens
        context_size (int): start context length

    Returns:
        _type_: _description_
    """
    for i in range(max_new_tokens):
        # logger.info(f"generate text step: {i}")
        # logger.info(f"{25 * '-'}")

        # Crop current context if it exceeds the supported context size
        # logger.info(f"token_idx before crop: {token_idx}")
        idx_cond = token_idx[:, -context_size:]
        # logger.info(f"token_idx after crop: {idx_cond}")
        
        # Get the predictions
        with torch.no_grad():
            logits = model(idx_cond)
        # logger.info(f"logits: \n{logits}")
        
        # Focus only on the last time step
        # shape: (batch, n_tokens, vocab_size) -> (batch, vocab_size)
        logits = logits[:, -1, :]
        # logger.info(f"logits: \n{logits}")
        
        # Softmax
        # shape: (batch, vocab_size)
        # probas = torch.softmax(logits, dim=-1)
        # logger.info(f"probas: \n{probas}, \nprobas.shape{probas.shape}")
        
        # Get the idx of the vocab entry with the highest probability value
        # shape: (batch, 1)
        idx_next = torch.argmax(logits, dim = -1, keepdim = True)
        # logger.info(f"idx_next: {idx_next}")
        
        # Append sampled index to the running sequence
        # shape: (batch, n_tokens+1)
        token_idx = torch.cat((token_idx, idx_next), dim = 1)
        # logger.info(f"token_idx: {token_idx}\n")

    return token_idx


def generate_simple_V1(
        model, 
        token_idx: torch.tensor, 
        max_new_tokens: int, 
        context_size: int
    ):
    """
    generate text

    Args:
        model (_type_): LLM model
        token_idx (torch.tensor): token_idx is array of indices in the current contex. shape: (batch, n_tokens)
        max_new_tokens (int): maximum length new tokens
        context_size (int): start context length

    Returns:
        _type_: _description_
    """
    for i in range(max_new_tokens):
        # logger.info(f"generate text step: {i}")
        # logger.info(f"{25 * '-'}")

        # Crop current context if it exceeds the supported context size
        # logger.info(f"token_idx before crop: {token_idx}")
        idx_cond = token_idx[:, -context_size:]
        # logger.info(f"token_idx after crop: {idx_cond}")
        
        # Get the predictions
        with torch.no_grad():
            logits = model(idx_cond)
        # logger.info(f"logits: \n{logits}")
        
        # Focus only on the last time step
        # shape: (batch, n_tokens, vocab_size) -> (batch, vocab_size)
        logits = logits[:, -1, :]
        # logger.info(f"logits: \n{logits}")
        
        # Softmax
        # shape: (batch, vocab_size)
        probas = torch.softmax(logits, dim=-1)
        # logger.info(f"probas: \n{probas}, \nprobas.shape{probas.shape}")
        
        # Get the idx of the vocab entry with the highest probability value
        # shape: (batch, 1)
        idx_next = torch.argmax(probas, dim = -1, keepdim = True)
        # logger.info(f"idx_next: {idx_next}")
        
        # Append sampled index to the running sequence
        # shape: (batch, n_tokens+1)
        token_idx = torch.cat((token_idx, idx_next), dim = 1)
        # logger.info(f"token_idx: {token_idx}\n")

    return token_idx


def generate(
        model, 
        token_idx: torch.tensor, 
        max_new_tokens: int,  
        context_size: int, 
        temperature: float=0.0, 
        top_k: float=None, 
        eos_id: int=None,
    ):
    """
    get logits, and only focus on last time step
    """
    for i in range(max_new_tokens):
        # logger.info(f"generate text step: {i}")
        # logger.info(f"{25 * '-'}")

        # crop current context if it exceeds the supported context size
        # logger.info(f"token_idx before crop: {token_idx}")
        idx_cond = token_idx[:, -context_size:]
        # logger.info(f"token_idx after crop: {idx_cond}")
        # get the predictions
        with torch.no_grad():
            logits = model(idx_cond)
        # logger.info(f"logits: {logits}")
        # focus only on the last time step
        # shape: (batch, n_tokens, vocab_size) -> (batch, vocab_size)
        logits = logits[:, -1, :]
        # logger.info(f"logits: \n{logits}")

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
        token_idx = torch.cat([token_idx, idx_next], dim=1)  # (batch_size, num_tokens+1)

    return token_idx




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
