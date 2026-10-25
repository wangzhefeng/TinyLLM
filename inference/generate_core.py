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

# python libraries
from pathlib import Path

import torch

from layers.sampling import sample_next_token
from utils.log_util import logger

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


def _resolve_ctx_len(model, context_length):
    """Resolve the effective context length for generation.

    兼容两类模型：
    - 绝对位置嵌入模型（GPT-2）：有 ``pos_embed`` 属性；
    - RoPE 模型（Llama/Qwen3/Gemma3）：没有 ``pos_embed``，取 ``cfg.context_length``。
    """
    if context_length:
        return context_length
    if hasattr(model, "pos_embed"):
        return model.pos_embed.num_embeddings
    cfg = getattr(model, "cfg", None)
    if cfg is not None and getattr(cfg, "context_length", None):
        return cfg.context_length
    raise ValueError(
        "context_length is None and the model exposes neither "
        "`pos_embed` nor `cfg.context_length`; please pass context_length explicitly."
    )


@torch.inference_mode()
def generate_simple(model, token_ids: torch.tensor, max_new_tokens: int, context_length: int):
    """
    generate text

    Args:
        model (_type_): LLM model
        token_ids (torch.tensor): token_ids is array of indices in the current contex. shape: (batch_ num_tokens)
        max_new_tokens (int): maximum length new tokens
        context_size (int): start context length
    """
    for _ in range(max_new_tokens):
        # Crop current context if it exceeds the supported context size
        cond_idx = token_ids[:, -context_length:]
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
        token_ids = torch.cat((token_ids, next_idx), dim=1)

    return token_ids


@torch.inference_mode()
def generate_simple_cached(model, token_ids: torch.tensor, max_new_tokens: int, context_length: int, use_cache=True):
    """
    generate text

    Args:
        model (_type_): LLM model
        token_ids (torch.tensor): token_ids is array of indices in the current contex. shape: (batch_ num_tokens)
        max_new_tokens (int): maximum length new tokens
        context_size (int): start context length
    """
    model.eval()
    ctx_len = _resolve_ctx_len(model, context_length)
    with torch.no_grad():
        if use_cache:
            # Init cache with full prompt
            model.reset_kv_cache()
            # Crop current context if it exceeds the supported context size
            cond_idx = token_ids[:, -ctx_len:]
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
                token_ids = torch.cat((token_ids, next_idx), dim=1)
                # c) feed model only the new token
                # -------------------
                logits = model(next_idx, use_cache)
        else:
            for _ in range(max_new_tokens):
                # Crop current context if it exceeds the supported context size
                cond_idx = token_ids[:, -ctx_len:]
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
                token_ids = torch.cat((token_ids, next_idx), dim=1)

    return token_ids


@torch.inference_mode()
def generate(model, token_ids: torch.tensor, max_new_tokens: int, context_length: int,
             temperature: float=0.0, top_k: int=None, top_p: float=None, eos_token_id: int=None, use_cache=False):
    model.eval()
    ctx_len = _resolve_ctx_len(model, context_length)
    finished = torch.zeros(token_ids.size(0), dtype=torch.bool, device=token_ids.device)
    cache = None
    if use_cache:
        model.reset_kv_cache()
        if hasattr(model, "pos_embed"):
            from layers.kv_cache import KVCache
            cache = KVCache(n_layers=len(model.trf_blocks))
        cond_idx = token_ids[:, -ctx_len:]
        logits = model(cond_idx, cache=cache) if cache is not None else model(cond_idx, use_cache)
    for _ in range(max_new_tokens):
        if not use_cache:
            logits = model(token_ids[:, -ctx_len:])
        next_idx = sample_next_token(
            logits[:, -1], temperature=temperature, top_k=top_k, top_p=top_p
        )
        if eos_token_id is not None:
            finished = finished | (next_idx.squeeze(1) == eos_token_id)
            if torch.all(finished):
                break
            next_idx = next_idx.masked_fill(finished.unsqueeze(1), eos_token_id)
        token_ids = torch.cat([token_ids, next_idx], dim=1)
        if use_cache:
            logits = model(next_idx, cache=cache) if cache is not None else model(next_idx, use_cache)
    return token_ids


@torch.inference_mode()
def generate_qwen3(model, token_ids: torch.tensor, max_new_tokens: int, context_length: int,
                   temperature: float=0.0, top_k: int=None, top_p: float=None, eos_token_id: int=None, use_cache=False):
    """
    get logits, and only focus on last time step

    Args:
        model (_type_): LLM model
        token_ids (torch.tensor): token_ids is array of indices in the current contex. shape: (batch_size, num_tokens)
        max_new_tokens (int): maximum length new tokens
        context_size (int): start context length
    """
    model.eval()
    input_len = token_ids.shape[1]
    ctx_len = _resolve_ctx_len(model, context_length)
    finished = torch.zeros(token_ids.size(0), dtype=torch.bool, device=token_ids.device)
    cache = None
    if use_cache:
        from layers.kv_cache import KVCache
        cache = KVCache(n_layers=model.cfg.n_layers)
        model.reset_kv_cache()
        logits = model(token_ids[:, -ctx_len:], cache=cache)
    for _ in range(max_new_tokens):
        if not use_cache:
            logits = model(token_ids[:, -ctx_len:])
        next_idx = sample_next_token(
            logits[:, -1], temperature=temperature, top_k=top_k, top_p=top_p
        )
        if eos_token_id is not None:
            finished = finished | (next_idx.squeeze(1) == eos_token_id)
            if torch.all(finished):
                break
            next_idx = next_idx.masked_fill(finished.unsqueeze(1), eos_token_id)
        token_ids = torch.cat([token_ids, next_idx], dim=1)
        if use_cache:
            logits = model(next_idx, cache=cache)
    return token_ids[:, input_len:]


@torch.inference_mode()
def generate_qwen3_stream(model, token_ids: torch.tensor, max_new_tokens: int, context_length: int,
                          temperature: float=0.0, top_k: int=None, top_p: float=None, eos_token_id: int=None, use_cache=False):
    """
    get logits, and only focus on last time step

    Args:
        model (_type_): LLM model
        token_ids (torch.tensor): token_ids is array of indices in the current contex. shape: (batch_size, num_tokens)
        max_new_tokens (int): maximum length new tokens
        context_size (int): start context length
    """
    model.eval()
    ctx_len = _resolve_ctx_len(model, context_length)
    with torch.no_grad():
        if use_cache:
            # Init cache with full prompt
            from layers.kv_cache import KVCache
            cache = KVCache(n_layers=model.cfg.n_layers)
            model.reset_kv_cache()
            # Crop current context if it exceeds the supported context size
            cond_idx = token_ids[:, -ctx_len:]
            # Get the predictions
            logits = model(cond_idx, cache=cache)
            for _ in range(max_new_tokens):
                # Focus only on the last time step, shape:(batch_size,num_tokens,vocab_size)->(batch_size,vocab_size)
                logits = logits[:, -1]
                next_idx = sample_next_token(logits, temperature=temperature, top_k=top_k, top_p=top_p)
                # Stop generating early if end-of-sequence token is encountered and eos_id is specified
                if eos_token_id is not None and torch.all(next_idx == eos_token_id):
                    break
                # stream
                yield next_idx
                # Append sampled index to the running sequence
                token_ids = torch.cat([token_ids, next_idx], dim=1)        # shape: (batch_size, num_tokens+1)
                # Feed model only the new token
                logits = model(next_idx, cache=cache)
        else:
            for _ in range(max_new_tokens):
                # Crop current context if it exceeds the supported context size
                cond_idx = token_ids[:, -ctx_len:]
                # Get the predictions
                logits = model(cond_idx)
                # Focus only on the last time step, shape:(batch_size,num_tokens,vocab_size)->(batch_size,vocab_size)
                logits = logits[:, -1]
                next_idx = sample_next_token(logits, temperature=temperature, top_k=top_k, top_p=top_p)
                # Stop generating early if end-of-sequence token is encountered and eos_id is specified
                if eos_token_id is not None and torch.all(next_idx == eos_token_id):
                    break
                # stream
                yield next_idx
                # Append sampled index to the running sequence
                token_ids = torch.cat([token_ids, next_idx], dim=1)        # shape: (batch_size, num_tokens+1)


@torch.inference_mode()
def generate_qwen3_optimized(model, token_ids: torch.tensor, max_new_tokens: int, context_length: int,
                             temperature: float=0.0, top_k: int=None, top_p: float=None, eos_token_id: int=None, use_cache=False):
    """
    get logits, and only focus on last time step

    Args:
        model (_type_): LLM model
        token_ids (torch.tensor): token_ids is array of indices in the current contex. shape: (batch_size, num_tokens)
        max_new_tokens (int): maximum length new tokens
        context_size (int): start context length
    """
    model.eval()
    ctx_len = _resolve_ctx_len(model, context_length)
    with torch.no_grad():
        if use_cache:
            # input length
            input_len = token_ids.shape[1]
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
            cond_idx = token_ids[:, -ctx_len:]
            # Get the predictions
            logits = model(cond_idx, cache=cache)
            for _ in range(max_new_tokens):
                # Focus only on the last time step, shape:(batch_size,num_tokens,vocab_size)->(batch_size,vocab_size)
                logits = logits[:, -1]
                next_idx = sample_next_token(logits, temperature=temperature, top_k=top_k, top_p=top_p)
                # Stop generating early if end-of-sequence token is encountered and eos_id is specified
                if eos_token_id is not None and torch.all(next_idx == eos_token_id):
                    break

                yield next_idx

                # Append sampled index to the running sequence
                token_ids = torch.cat([token_ids, next_idx], dim=1)        # shape: (batch_size, num_tokens+1)
                # Feed model only the new token
                logits = model(next_idx, cache=cache)
            # return token_ids[:, -1][:, input_len:]
        else:
            # input length
            input_len = token_ids.shape[1]
            for _ in range(max_new_tokens):
                # Crop current context if it exceeds the supported context size
                cond_idx = token_ids[:, -ctx_len:]
                # Get the predictions
                logits = model(cond_idx)
                # Focus only on the last time step, shape:(batch_size,num_tokens,vocab_size)->(batch_size,vocab_size)
                logits = logits[:, -1]
                next_idx = sample_next_token(logits, temperature=temperature, top_k=top_k, top_p=top_p)
                # Stop generating early if end-of-sequence token is encountered and eos_id is specified
                if eos_token_id is not None and torch.all(next_idx == eos_token_id):
                    break

                yield next_idx

                # Append sampled index to the running sequence
                token_ids = torch.cat([token_ids, next_idx], dim=1)        # shape: (batch_size, num_tokens+1)
            # return token_ids[:, input_len:]


def _prefill_last_logits(logits_full, attn_mask, batch_size):
    """Prefill 后取各行的"下一个 token" logits。

    右填充 batch 中短行的最后位置是 pad 位，统一取 [:, -1] 会让首 token
    从 pad 位置预测；按 attn_mask 各行真实末位 gather 保证与逐条生成一致。
    """
    if attn_mask is not None:
        real_last = attn_mask.long().sum(dim=1) - 1
        return logits_full[torch.arange(batch_size, device=logits_full.device), real_last]
    return logits_full[:, -1]


@torch.inference_mode()
def generate_qwen3_batched_cache(model, token_ids: torch.tensor, max_new_tokens: int, context_length: int,
                                 temperature: float=0.0, top_k: int=None, top_p: float=None, eos_token_id: int=None,
                                 attn_mask=None, pad_id=None):
    """
    get logits, and only focus on last time step

    Args:
        model (_type_): LLM model
        token_ids (torch.tensor): token_ids is array of indices in the current contex. shape: (batch_size, num_tokens)
        max_new_tokens (int): maximum length new tokens
        context_size (int): start context length
    """
    model.eval()
    ctx_len = _resolve_ctx_len(model, context_length)

    device = token_ids.device
    batch_size, input_len = token_ids.shape

    if attn_mask is None and pad_id is not None:
        attn_mask = (token_ids != pad_id).to(torch.bool)
    if attn_mask is not None:
        attn_mask = attn_mask.to(torch.bool).to(device)

    with torch.no_grad():
        # Init cache and model position
        from layers.kv_cache import KVCache
        cache = KVCache(n_layers=model.cfg.n_layers)
        # Crop current context if it exceeds the supported context size
        cond_idx = token_ids[:, -ctx_len:]
        # Prefill(Get the predictions)
        logits = _prefill_last_logits(model(cond_idx, cache=cache, attention_mask=(None if attn_mask is None else ~attn_mask.bool())), attn_mask, batch_size)
        # Track which sequences have already producted EOS
        if eos_token_id is not None:
            # If a prompt already ends with EOS, consider it finished
            finished = (token_ids[:, -1] == eos_token_id)
        else:
            finished = None
        # Decode
        cur_attn = attn_mask
        for _ in range(max_new_tokens):
            # If all sequences are already finished, stop
            if eos_token_id is not None and finished is not None and torch.all(finished):
                break
            next_idx = sample_next_token(logits, temperature=temperature, top_k=top_k, top_p=top_p)  # shape:(batch_size, 1)
            # Force already finished rows to keep emitting EOS to maintain shape
            if eos_token_id is not None:
                eos_tok = next_idx.new_full((batch_size, 1), eos_token_id)
                next_idx = torch.where(finished.view(batch_size, 1), eos_tok, next_idx)
            # Extend mask to include the newly generated token
            if cur_attn is not None:
                ones = torch.ones((batch_size, 1), dtype=cur_attn.dtype, device=device)
                cur_attn = torch.cat([cur_attn, ones], dim=1)
            # Advance one token with KV cache
            logits = model(next_idx, cache=cache, attention_mask=(None if cur_attn is None else ~cur_attn.bool()))[:, -1]
            token_ids = torch.cat([token_ids, next_idx], dim=1)
            # Update finished mask after appending this step's token
            if eos_token_id is not None:
                finished = finished | (next_idx.squeeze(1) == eos_token_id)

        return token_ids[:, input_len:]


@torch.inference_mode()
def generate_qwen3_batched_stream_cache(model, token_ids: torch.tensor, max_new_tokens: int, context_length: int,
                                        temperature: float=0.0, top_k: int=None, top_p: float=None, eos_token_id: int=None,
                                        attn_mask=None, pad_id=None):
    """
    get logits, and only focus on last time step

    Args:
        model (_type_): LLM model
        token_ids (torch.tensor): token_ids is array of indices in the current contex. shape: (batch_size, num_tokens)
        max_new_tokens (int): maximum length new tokens
        context_size (int): start context length
    """
    model.eval()
    ctx_len = _resolve_ctx_len(model, context_length)

    device = token_ids.device
    batch_size, input_len = token_ids.shape

    if attn_mask is None and pad_id is not None:
        attn_mask = (token_ids != pad_id).to(torch.bool)
    if attn_mask is not None:
        attn_mask = attn_mask.to(torch.bool).to(device)

    with torch.no_grad():
        # Init cache and model position
        from layers.kv_cache import KVCache
        cache = KVCache(n_layers=model.cfg.n_layers)
        # Crop current context if it exceeds the supported context size
        cond_idx = token_ids[:, -ctx_len:]
        # Prefill(Get the predictions)
        logits = _prefill_last_logits(model(cond_idx, cache=cache, attention_mask=(None if attn_mask is None else ~attn_mask.bool())), attn_mask, batch_size)
        # Decode
        cur_attn = attn_mask
        for _ in range(max_new_tokens):
            next_idx = sample_next_token(logits, temperature=temperature, top_k=top_k, top_p=top_p)  # shape:(batch_size, 1)
            # If all sequences are already finished, stop
            if eos_token_id is not None and torch.all(next_idx.squeeze(-1) == eos_token_id):
                break

            yield next_idx

            # Extend mask to include the newly generated token
            if cur_attn is not None:
                ones = torch.ones((batch_size, 1), dtype=cur_attn.dtype, device=device)
                cur_attn = torch.cat([cur_attn, ones], dim=1)
            # Advance one token with KV cache
            logits = model(next_idx, cache=cache, attention_mask=(None if cur_attn is None else ~cur_attn.bool()))[:, -1]
            token_ids = torch.cat([token_ids, next_idx], dim=1)


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
def generate_text_basic_batched_cache_stop(model, token_ids: torch.tensor, max_new_tokens: int, context_length: int,
                                           temperature: float=0.0, top_k: int=None, top_p: float=None, eos_token_id=None,
                                           attn_mask=None, pad_id=None):
    """
    Same as generate_text_basic_batched_cache but
    with per-sequence early stop.
    I.e., finished rows that see an EOS written don't
    participate in forward pass anymore.
    """
    model.eval()
    ctx_len = _resolve_ctx_len(model, context_length)

    device = token_ids.device
    batch_size, input_len = token_ids.shape

    # Build attention mask
    if attn_mask is None and pad_id is not None:
        attn_mask = (token_ids != pad_id)
    if attn_mask is not None:
        attn_mask = attn_mask.to(torch.bool).to(device)
    with torch.no_grad():
        # Init cache and prefill once on full batch
        from layers.kv_cache import KVCache
        cache = KVCache(n_layers=model.cfg.n_layers)
        # Crop current context if it exceeds the supported context size
        cond_idx = token_ids[:, -ctx_len:]
        # Prefill(Get the predictions)
        logits = _prefill_last_logits(model(cond_idx, cache=cache, attention_mask=(None if attn_mask is None else ~attn_mask.bool())), attn_mask, batch_size)  # (B, V)

        finished_full = torch.zeros(batch_size, dtype=torch.bool, device=device)
        active_idx = torch.arange(batch_size, device=device)  # active rows -> original rows
        cur_attn_active = attn_mask                  # mirrors the active cache
        generated_full_steps = []                    # list of (B,1) step tensors
        for _ in range(max_new_tokens):
            # Next tokens for the active sub-batch
            next_token_active = sample_next_token(logits, temperature=temperature, top_k=top_k, top_p=top_p)  # (B_active, 1)

            # Scatter into a full-sized (B,1) step tensor (EOS for finished rows)
            fill_val = int(eos_token_id) if eos_token_id is not None else 0
            step_full = torch.full((batch_size, 1), fill_value=fill_val, dtype=token_ids.dtype, device=device)
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
            logits = model(next_token_survivors, cache=cache, attention_mask=~cur_attn_active.bool())[:, -1]

        # Concatenate per-step tensors; return only the generated part
        if generated_full_steps:
            return torch.cat(generated_full_steps, dim=1)  # (B, L_generated)
        else:
            return torch.empty((batch_size, 0), dtype=token_ids.dtype, device=device)


@torch.inference_mode()
def generate_text_basic_batched_stream_cache_stop(model, token_ids: torch.Tensor, max_new_tokens: int, context_length: int,
                                                  temperature: float = 0.0, top_k: int | None = None, top_p: float | None = None, eos_token_id: int | None = None,
                                                  attn_mask: torch.Tensor | None = None, pad_id: int | None = None):
    """
    Same as generate_text_basic_batched_stream_cache but
    with per-sequence early stop.
    """
    model.eval()
    ctx_len = _resolve_ctx_len(model, context_length)

    device = token_ids.device
    batch_size, input_len = token_ids.shape

    if attn_mask is None and pad_id is not None:
        attn_mask = (token_ids != pad_id)
    if attn_mask is not None:
        attn_mask = attn_mask.to(torch.bool).to(device)

    with torch.no_grad():
        # Init cache and model position
        from layers.kv_cache import KVCache
        cache = KVCache(n_layers=model.cfg.n_layers)
        # Crop current context if it exceeds the supported context size
        cond_idx = token_ids[:, -ctx_len:]
        # Prefill(Get the predictions)
        logits = _prefill_last_logits(model(cond_idx, cache=cache, attention_mask=(None if attn_mask is None else ~attn_mask.bool())), attn_mask, batch_size)  # (B, V)

        finished_full = torch.zeros(batch_size, dtype=torch.bool, device=device)
        active_idx = torch.arange(batch_size, device=device)
        cur_attn_active = attn_mask
        for _ in range(max_new_tokens):
            next_token_active = sample_next_token(logits, temperature=temperature, top_k=top_k, top_p=top_p)  # (B_active, 1)

            # Build full-sized step to yield
            fill_val = int(eos_token_id) if eos_token_id is not None else 0
            step_full = torch.full(
                (batch_size, 1), fill_value=fill_val,dtype=token_ids.dtype, device=device
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

            logits = model(next_token_survivors, cache=cache, attention_mask=~cur_attn_active.bool())[:, -1]


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


def repetition_penalty(logits: torch.Tensor, token_ids: torch.Tensor, penalty: float = 1.2) -> torch.Tensor:
    """CTRL 式重复惩罚（https://arxiv.org/abs/1909.05858）。

    对已生成过的 token：正 logit 除以 penalty，负 logit 乘以 penalty，
    从而降低（或反向提升）已出现 token 的概率。penalty=1.0 时无操作。

    Args:
        logits: (batch_size, vocab_size)
        token_ids: (batch_size, T) 已生成的 token（含 prompt）
        penalty: >1 抑制重复；<1 鼓励重复

    Returns:
        (batch_size, vocab_size) 修正后的 logits（新张量，不改入参）
    """
    if penalty == 1.0 or token_ids is None or token_ids.numel() == 0:
        return logits
    vocab_size = logits.size(-1)
    seen = torch.zeros_like(logits, dtype=torch.bool)
    seen.scatter_(1, token_ids.long(), True)                 # (B, V) 标记已出现
    logits = logits.clone()
    pos = logits > 0
    logits = torch.where(seen & pos, logits / penalty, logits)
    logits = torch.where(seen & ~pos, logits * penalty, logits)
    return logits


@torch.inference_mode()
def beam_search_generate(model, token_ids: torch.Tensor, max_new_tokens: int = 20,
                         num_beams: int = 3, context_length: int | None = None,
                         length_penalty: float = 1.0, eos_token_id: int | None = None,
                         repetition_pen: float = 1.0, return_beams: bool = False,
                         beam_size: int | None = None):
    """Beam search 解码（教学实现，batch=1，无 KV cache）。

    算法要点：
    - 每步保留累计 log-prob 之和最高的 ``num_beams`` 条候选（多 beam 展开）；
    - 长度惩罚：排序分数 = logprob 总和 / gen_len ** length_penalty，
      gen_len 为该 beam 实际生成的 token 数（含 EOS；冻结后填充的 EOS 不计入）；
    - EOS 冻结：某 beam 生成 eos 后冻结为完成候选，不再扩展——实现上只允许
      延续 EOS 且累计分不变（填充仅为保持张量等宽，不参与打分与长度）；
    - 停止条件：达到 max_new_tokens，或全部 beam 完成；
    - 最终按归一化分数取最优 beam。

    为什么不用 KV cache：beam 每步会在候选间分叉/回溯，cache 需要 按
    beam 复制、重排与回滚，教学实现为了逻辑清晰选择每步整段重算前缀
    （代价 O(T^2) 前向，beam 数多时更明显）。

    Args:
        model: forward(x) 返回 (B, T, V) logits 的模型
        token_ids: (1, T) prompt（batch 目前仅支持 1）
        num_beams: beam 宽度；num_beams=1 且 length_penalty=0 时退化为 greedy
        context_length: 模型支持的最大上下文长度；None 时自动推断
        length_penalty: 长度归一化指数（0=只比 logprob 总和，1.0=按长度平均）
        eos_token_id: 结束 token id；None 表示不启用提前停止
        repetition_pen: 传给 repetition_penalty 的惩罚系数
        return_beams: True 时返回全部 beam（num_beams, T+L），按分数降序
        beam_size: ``num_beams`` 的旧别名（向后兼容），显式传入时优先于 num_beams

    Returns:
        (1, T+L) 最优序列（含 prompt）；return_beams=True 时为 (num_beams, T+L)
    """
    if beam_size is not None:
        num_beams = beam_size
    if num_beams < 1:
        raise ValueError("num_beams must be >= 1")
    model.eval()
    ctx_len = _resolve_ctx_len(model, context_length)
    with torch.no_grad():
        prompt = token_ids[:, -ctx_len:]
        assert prompt.size(0) == 1, "beam search 教学实现要求 batch=1"
        device = prompt.device
        beams = prompt.repeat(num_beams, 1)                  # (num_beams, T)
        scores = torch.full((num_beams,), float("-inf"), device=device)
        scores[0] = 0.0                                      # 初始只有第 0 条有效
        beam_lens = torch.zeros(num_beams, dtype=torch.long, device=device)
        finished = torch.zeros(num_beams, dtype=torch.bool, device=device)

        for _ in range(max_new_tokens):
            logits = model(beams)[:, -1, :]                  # (num_beams, V)
            if repetition_pen != 1.0:
                logits = repetition_penalty(logits, beams, repetition_pen)
            logprobs = torch.log_softmax(logits.float(), dim=-1)
            if eos_token_id is not None:
                # 冻结已完成的 beam：只能延续 EOS 且不改变累计分
                logprobs[finished] = float("-inf")
                logprobs[finished, eos_token_id] = 0.0
            cand = scores.unsqueeze(1) + logprobs           # (num_beams, V)
            top_scores, top_idx = cand.view(-1).topk(num_beams)
            vocab_size = logprobs.size(-1)
            beam_idx = torch.div(top_idx, vocab_size, rounding_mode="floor")
            token_idx = top_idx % vocab_size
            was_finished = finished[beam_idx]
            beams = torch.cat([beams[beam_idx], token_idx.unsqueeze(1)], dim=1)
            scores = top_scores
            # 冻结 beam 延续的 EOS 只是填充，不计入已生成长度
            beam_lens = beam_lens[beam_idx] + (~was_finished).long()
            if eos_token_id is not None:
                finished = was_finished | (token_idx == eos_token_id)
                if bool(finished.all()):
                    break

        # 结束时按归一化分数排序：logprob 总和 / gen_len ** length_penalty
        gen_len = beam_lens.clamp(min=1).to(scores.dtype)
        normed = scores / gen_len.pow(length_penalty)
        order = normed.argsort(descending=True)
        if return_beams:
            return beams[order]
        return beams[order[:1]]




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
