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

from utils.log_util import logger

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


@torch.inference_mode()
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


@torch.inference_mode()
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


@torch.inference_mode()
def generate(model, token_idx: torch.tensor, max_new_tokens: int, context_length: int, 
             temperature: float=0.0, top_k: int=None, eos_token_id: int=None, use_cache=False):
    """
    get logits, and only focus on last time step

    Args:
        model (_type_): LLM model
        token_idx (torch.tensor): token_idx is array of indices in the current contex. shape: (batch_size, num_tokens)
        max_new_tokens (int): maximum length new tokens
        context_size (int): start context length
    """
    model.eval()
    input_len = token_idx.shape[1]
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
                if eos_token_id is not None and torch.all(next_idx == eos_token_id):
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
                # logits = model(cond_idx, use_cache=False)
                logits = model(cond_idx)
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
                if eos_token_id is not None and torch.all(next_idx == eos_token_id):
                    break
                
                # TODO yield next_idx

                # Append sampled index to the running sequence
                token_idx = torch.cat([token_idx, next_idx], dim=1)        # shape: (batch_size, num_tokens+1)

    return token_idx


@torch.inference_mode()
def generate_qwen3(model, token_idx: torch.tensor, max_new_tokens: int, context_length: int, 
                   temperature: float=0.0, top_k: int=None, eos_token_id: int=None, use_cache=False):
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
            # input length
            input_len = token_idx.shape[1]
            # Init cache with full prompt
            from layers.kv_cache import KVCache
            cache = KVCache(n_layers=model.cfg.n_layers)
            model.reset_kv_cache()
            # Crop current context if it exceeds the supported context size
            cond_idx = token_idx[:, -ctx_len:]
            # Get the predictions
            logits = model(cond_idx, cache=cache)
            for _ in range(max_new_tokens):
                # Focus only on the last time step, shape:(batch_size,num_tokens,vocab_size)->(batch_size,vocab_size)
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
                if eos_token_id is not None and torch.all(next_idx == eos_token_id):
                    break
                # Append sampled index to the running sequence
                token_idx = torch.cat([token_idx, next_idx], dim=1)        # shape: (batch_size, num_tokens+1)
                # Feed model only the new token
                logits = model(next_idx, cache=cache)
            return token_idx[:, -1][:, input_len:]
        else:
            # input length
            input_len = token_idx.shape[1]
            for _ in range(max_new_tokens):
                # Crop current context if it exceeds the supported context size
                cond_idx = token_idx[:, -ctx_len:]
                # Get the predictions
                logits = model(cond_idx)
                # Focus only on the last time step, shape:(batch_size,num_tokens,vocab_size)->(batch_size,vocab_size)
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
                if eos_token_id is not None and torch.all(next_idx == eos_token_id):
                    break
                # Append sampled index to the running sequence
                token_idx = torch.cat([token_idx, next_idx], dim=1)        # shape: (batch_size, num_tokens+1)
            return token_idx[:, input_len:]


@torch.inference_mode()
def generate_qwen3_stream(model, token_idx: torch.tensor, max_new_tokens: int, context_length: int, 
                          temperature: float=0.0, top_k: int=None, eos_token_id: int=None, use_cache=False):
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
            from layers.kv_cache import KVCache
            cache = KVCache(n_layers=model.cfg.n_layers)
            model.reset_kv_cache()
            # Crop current context if it exceeds the supported context size
            cond_idx = token_idx[:, -ctx_len:]
            # Get the predictions
            logits = model(cond_idx, cache=cache)
            for _ in range(max_new_tokens):
                # Focus only on the last time step, shape:(batch_size,num_tokens,vocab_size)->(batch_size,vocab_size)
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
                if eos_token_id is not None and torch.all(next_idx == eos_token_id):
                    break
                # stream
                yield next_idx
                # Append sampled index to the running sequence
                token_idx = torch.cat([token_idx, next_idx], dim=1)        # shape: (batch_size, num_tokens+1)
                # Feed model only the new token
                logits = model(next_idx, cache=cache)
        else:
            for _ in range(max_new_tokens):
                # Crop current context if it exceeds the supported context size
                cond_idx = token_idx[:, -ctx_len:]
                # Get the predictions
                logits = model(cond_idx)
                # Focus only on the last time step, shape:(batch_size,num_tokens,vocab_size)->(batch_size,vocab_size)
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
                if eos_token_id is not None and torch.all(next_idx == eos_token_id):
                    break
                # stream
                yield next_idx
                # Append sampled index to the running sequence
                token_idx = torch.cat([token_idx, next_idx], dim=1)        # shape: (batch_size, num_tokens+1)


@torch.inference_mode()
def generate_qwen3_optimized(model, token_idx: torch.tensor, max_new_tokens: int, context_length: int, 
                             temperature: float=0.0, top_k: int=None, eos_token_id: int=None, use_cache=False):
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
            # input length
            input_len = token_idx.shape[1]
            # Init cache with full prompt
            from layers.kv_cache import KVCache_optimized
            cache = KVCache_optimized(
                n_layers=model.cfg.n_layers,
                max_len=model.cfg.context_length,
                num_kv_groups=model.cfg.n_kv_groups,
                head_dim=model.cfg.head_dim,
                device=next(model.parameters()).device,
                dtype=model.cfg.dtype
            )
            model.reset_kv_cache()
            # Crop current context if it exceeds the supported context size
            cond_idx = token_idx[:, -ctx_len:]
            # Get the predictions
            logits = model(cond_idx, cache=cache)[:, -1]
            for _ in range(max_new_tokens):
                # Focus only on the last time step, shape:(batch_size,num_tokens,vocab_size)->(batch_size,vocab_size)
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
                if eos_token_id is not None and torch.all(next_idx == eos_token_id):
                    break

                yield next_idx

                # Append sampled index to the running sequence
                token_idx = torch.cat([token_idx, next_idx], dim=1)        # shape: (batch_size, num_tokens+1)
                # Feed model only the new token
                logits = model(next_idx, cache=cache)
            # return token_idx[:, -1][:, input_len:]
        else:
            # input length
            input_len = token_idx.shape[1]
            for _ in range(max_new_tokens):
                # Crop current context if it exceeds the supported context size
                cond_idx = token_idx[:, -ctx_len:]
                # Get the predictions
                logits = model(cond_idx)
                # Focus only on the last time step, shape:(batch_size,num_tokens,vocab_size)->(batch_size,vocab_size)
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
                if eos_token_id is not None and torch.all(next_idx == eos_token_id):
                    break

                yield next_idx

                # Append sampled index to the running sequence
                token_idx = torch.cat([token_idx, next_idx], dim=1)        # shape: (batch_size, num_tokens+1)
            # return token_idx[:, input_len:]


@torch.inference_mode()
def generate_qwen3_batched_cache(model, token_idx: torch.tensor, max_new_tokens: int, context_length: int, 
                                 temperature: float=0.0, top_k: int=None, eos_token_id: int=None, 
                                 attn_mask=None, pad_id=None):
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
    
    device = token_idx.device
    batch_size, input_len = token_idx.shape
    
    if attn_mask is None and pad_id is not None:
        attn_mask = (token_idx != pad_id).to(torch.bool)
    if attn_mask is not None:
        attn_mask = attn_mask.to(torch.bool).to(device)
    
    with torch.no_grad():
        # Init cache and model position
        from layers.kv_cache import KVCache
        cache = KVCache(n_layers=model.cfg.n_layers)
        # Crop current context if it exceeds the supported context size
        cond_idx = token_idx[:, -ctx_len:]
        # Prefill(Get the predictions)
        logits = model(cond_idx, cache=cache, attn_mask=attn_mask)[:, -1]
        # Track which sequences have already producted EOS
        if eos_token_id is not None:
            # If a prompt already ends with EOS, consider it finished
            finished = (token_idx[:, -1] == eos_token_id)
        else:
            finished = None
        # Decode
        cur_attn = attn_mask
        for _ in range(max_new_tokens):
            # If all sequences are already finished, stop
            if eos_token_id is not None and finished is not None and torch.all(finished):
                break
            next_idx = torch.argmax(logits, dim=-1, keepdim=True)  # shape:(batch_size, 1) 
            # Force already finished rows to keep emitting EOS to maintain shape
            if eos_token_id is not None:
                eos_tok = next_idx.new_full((batch_size, 1), eos_token_id)
                next_idx = torch.where(finished.view(batch_size, 1), eos_tok, next_idx)
            # Extend mask to include the newly generated token
            if cur_attn is not None:
                ones = torch.ones((batch_size, 1), dtype=cur_attn.dtype, device=device)
                cur_attn = torch.cat([cur_attn, ones], dim=1)
            # Advance one token with KV cache
            logits = model(next_idx, cache=cache, attn_mask=cur_attn)[:, -1]
            token_idx = torch.cat([token_idx, next_idx], dim=1)
            # Update finished mask after appending this step's token
            if eos_token_id is not None:
                finished = finished | (next_idx.squeeze(1) == eos_token_id)
        
        return token_idx[:, input_len:]


@torch.inference_mode()
def generate_qwen3_batched_stream_cache(model, token_idx: torch.tensor, max_new_tokens: int, context_length: int, 
                                        temperature: float=0.0, top_k: int=None, eos_token_id: int=None, 
                                        attn_mask=None, pad_id=None):
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
    
    device = token_idx.device
    batch_size, input_len = token_idx.shape
    
    if attn_mask is None and pad_id is not None:
        attn_mask = (token_idx != pad_id).to(torch.bool)
    if attn_mask is not None:
        attn_mask = attn_mask.to(torch.bool).to(device)
    
    with torch.no_grad():
        # Init cache and model position
        from layers.kv_cache import KVCache
        cache = KVCache(n_layers=model.cfg.n_layers)
        # Crop current context if it exceeds the supported context size
        cond_idx = token_idx[:, -ctx_len:]
        # Prefill(Get the predictions)
        logits = model(cond_idx, cache=cache, attn_mask=attn_mask)[:, -1]
        # Decode
        cur_attn = attn_mask
        for _ in range(max_new_tokens):
            next_idx = torch.argmax(logits, dim=-1, keepdim=True)  # shape:(batch_size, 1) 
            # If all sequences are already finished, stop
            if eos_token_id is not None and torch.all(next_idx.squeeze(-1) == eos_token_id):
                break

            yield next_idx

            # Extend mask to include the newly generated token
            if cur_attn is not None:
                ones = torch.ones((batch_size, 1), dtype=cur_attn.dtype, device=device)
                cur_attn = torch.cat([cur_attn, ones], dim=1)
            # Advance one token with KV cache
            logits = model(next_idx, cache=cache, attn_mask=cur_attn)[:, -1]
            token_idx = torch.cat([token_idx, next_idx], dim=1)


def shrink_kv_cache_inplace(cache, keep_mask, n_layers):
    if keep_mask.dtype != torch.bool:
        keep_mask = keep_mask.to(torch.bool)
    for i in range(n_layers):
        kv = cache.get(i)
        if kv is None:
            continue
        K, V = kv
        K = K[keep_mask]  # shrink along batch dim
        V = V[keep_mask]
        cache.update(i, (K, V))


@torch.inference_mode()
def generate_text_basic_batched_cache_stop(model, token_idx: torch.tensor, max_new_tokens: int, context_length: int, 
                                           temperature: float=0.0, top_k: int=None, eos_token_id=None, 
                                           attn_mask=None, pad_id=None):
    """
    Same as generate_text_basic_batched_cache but
    with per-sequence early stop.
    I.e., finished rows that see an EOS written don't
    participate in forward pass anymore.
    """
    model.eval()
    ctx_len = context_length or model.pos_embed.num_embeddings

    device = token_idx.device
    batch_size, input_len = token_idx.shape

    # Build attention mask
    if attn_mask is None and pad_id is not None:
        attn_mask = (token_idx != pad_id)
    if attn_mask is not None:
        attn_mask = attn_mask.to(torch.bool).to(device)
    with torch.no_grad():
        # Init cache and prefill once on full batch
        from layers.kv_cache import KVCache
        cache = KVCache(n_layers=model.cfg.n_layers)
        # Crop current context if it exceeds the supported context size
        cond_idx = token_idx[:, -ctx_len:]
        # Prefill(Get the predictions)
        logits = model(cond_idx, cache=cache, attn_mask=attn_mask)[:, -1]  # (B, V)

        finished_full = torch.zeros(batch_size, dtype=torch.bool, device=device)
        active_idx = torch.arange(batch_size, device=device)  # active rows -> original rows
        cur_attn_active = attn_mask                  # mirrors the active cache
        generated_full_steps = []                    # list of (B,1) step tensors
        for _ in range(max_new_tokens):
            # Next tokens for the active sub-batch
            next_token_active = torch.argmax(logits, dim=-1, keepdim=True)  # (B_active, 1)

            # Scatter into a full-sized (B,1) step tensor (EOS for finished rows)
            fill_val = int(eos_token_id) if eos_token_id is not None else 0
            step_full = torch.full((batch_size, 1), fill_value=fill_val, dtype=token_idx.dtype, device=device)
            step_full.index_copy_(0, active_idx, next_token_active)
            generated_full_steps.append(step_full)

            # Update finished bookkeeping in full-batch coordinates
            if eos_token_id is not None:
                newly_finished_active = (next_token_active.squeeze(1) == eos_token_id)
                finished_full.index_put_(
                    (active_idx,),
                    newly_finished_active | finished_full.index_select(0, active_idx)
                )
            else:
                newly_finished_active = torch.zeros_like(
                    next_token_active.squeeze(1), dtype=torch.bool, device=device
                )

            if eos_token_id is not None and torch.all(finished_full):
                break

            # Keep only survivors in the compute batch
            keep_mask_active = ~newly_finished_active
            if keep_mask_active.ndim == 0:
                keep_any = bool(keep_mask_active.item())
            else:
                keep_any = bool(keep_mask_active.any().item())
            if not keep_any:
                break

            next_token_survivors = next_token_active[keep_mask_active]  # (B_surv, 1)
            active_idx = active_idx[keep_mask_active]

            # Shrink attn mask and append a "1" for the generated token
            if cur_attn_active is not None:
                cur_attn_active = cur_attn_active[keep_mask_active]
                ones = torch.ones((cur_attn_active.size(0), 1), dtype=cur_attn_active.dtype, device=device)
                cur_attn_active = torch.cat([cur_attn_active, ones], dim=1)

            # Shrink KV cache along batch dim to survivors
            shrink_kv_cache_inplace(cache, keep_mask_active, model.cfg.n_layers)

            # Advance one token for survivors only
            logits = model(next_token_survivors, cache=cache, attn_mask=cur_attn_active)[:, -1]

        # Concatenate per-step tensors; return only the generated part
        if generated_full_steps:
            return torch.cat(generated_full_steps, dim=1)  # (B, L_generated)
        else:
            return torch.empty((batch_size, 0), dtype=token_idx.dtype, device=device)


@torch.inference_mode()
def generate_text_basic_batched_stream_cache_stop(model, token_idx: torch.Tensor, max_new_tokens: int, context_length: int,
                                                  temperature: float = 0.0, top_k: int | None = None, eos_token_id: int | None = None, 
                                                  attn_mask: torch.Tensor | None = None, pad_id: int | None = None):
    """
    Same as generate_text_basic_batched_stream_cache but
    with per-sequence early stop.
    """
    model.eval()
    ctx_len = context_length or model.pos_embed.num_embeddings

    device = token_idx.device
    batch_batch, input_len = token_idx.shape

    if attn_mask is None and pad_id is not None:
        attn_mask = (token_idx != pad_id)
    if attn_mask is not None:
        attn_mask = attn_mask.to(torch.bool).to(device)
    
    with torch.no_grad():
        # Init cache and model position
        from layers.kv_cache import KVCache
        cache = KVCache(n_layers=model.cfg.n_layers)
        # Crop current context if it exceeds the supported context size
        cond_idx = token_idx[:, -ctx_len:]
        # Prefill(Get the predictions)
        logits = model(cond_idx, cache=cache, attn_mask=attn_mask)[:, -1]  # (B, V)

        finished_full = torch.zeros(batch_batch, dtype=torch.bool, device=device)
        active_idx = torch.arange(batch_batch, device=device)
        cur_attn_active = attn_mask
        for _ in range(max_new_tokens):
            next_token_active = torch.argmax(logits, dim=-1, keepdim=True)  # (B_active, 1)

            # Build full-sized step to yield
            fill_val = int(eos_token_id) if eos_token_id is not None else 0
            step_full = torch.full(
                (batch_batch, 1), fill_value=fill_val,dtype=token_idx.dtype, device=device
            )
            step_full.index_copy_(0, active_idx, next_token_active)

            if eos_token_id is not None:
                newly_finished_active = (next_token_active.squeeze(1) == eos_token_id)
                finished_full.index_put_(
                    (active_idx,),
                    newly_finished_active | finished_full.index_select(0, active_idx)
                )
            else:
                newly_finished_active = torch.zeros_like(
                    next_token_active.squeeze(1), dtype=torch.bool, device=device
                )

            # Yield before shrinking so callers still see exactly one (B,1) per step
            yield step_full

            if eos_token_id is not None and torch.all(finished_full):
                break

            keep_mask_active = ~newly_finished_active
            if keep_mask_active.ndim == 0:
                keep_any = bool(keep_mask_active.item())
            else:
                keep_any = bool(keep_mask_active.any().item())
            if not keep_any:
                break

            next_token_survivors = next_token_active[keep_mask_active]
            active_idx = active_idx[keep_mask_active]

            if cur_attn_active is not None:
                cur_attn_active = cur_attn_active[keep_mask_active]
                ones = torch.ones((cur_attn_active.size(0), 1), dtype=cur_attn_active.dtype, device=device)
                cur_attn_active = torch.cat([cur_attn_active, ones], dim=1)

            shrink_kv_cache_inplace(cache, keep_mask_active, model.cfg.n_layers)

            logits = model(next_token_survivors, cache=cache, attn_mask=cur_attn_active)[:, -1]


def generate_stats(output_token_ids, tokenizer, start_time, end_time):
    total_time = end_time - start_time
    logger.info(f"Time: {total_time:.2f} sec")
    logger.info(f"{int(output_token_ids.numel() / total_time)} tokens/sec")
    for name, backend in (("CUDA", getattr(torch, "cuda", None)), ("XPU", getattr(torch, "xpu", None))):
        if backend is not None and backend.is_available():
            max_mem_bytes = backend.max_memory_allocated()
            max_mem_gb = max_mem_bytes / (1024 ** 3)
            logger.info(f"Max {name} memory allocated: {max_mem_gb:.2f} GB")
            backend.reset_peak_memory_stats()
    # output_text = tokenizer.decode(output_token_ids.squeeze(0).tolist())
    # logger.info(f"\n{output_text}")




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
