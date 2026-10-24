from __future__ import annotations

__all__ = ["generate_text", "generate_text_stream"]


def _encode(prompt, family, tokenizer_model, tokenizer_file_path, device, **tokenizer_kwargs):
    import torch
    from layers.tokenizers.tokenization import choose_tokenizer

    if family == "qwen3":
        if tokenizer_file_path is None:
            raise ValueError("Qwen3 requires tokenizer_file_path")
        tokenizer_kwargs["tokenizer_file_path"] = tokenizer_file_path
    elif family == "gemma3":
        if tokenizer_file_path is None:
            raise ValueError("Gemma3 requires tokenizer_file_path")
        tokenizer_kwargs["tokenizer_file_path"] = tokenizer_file_path
    name = tokenizer_model or {
        "gpt2": "tiktoken_gpt2_bpe",
        "llama": "llama2",
    }.get(family, family)
    tokenizer = choose_tokenizer(name, **tokenizer_kwargs)
    token_ids = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
    return token_ids, tokenizer


def _context_length(model, context_length):
    if context_length is not None:
        return context_length
    if hasattr(model, "cfg"):
        return model.cfg.context_length
    return model.pos_embed.num_embeddings


def generate_text(
    model,
    prompt: str,
    family: str = "gpt2",
    tokenizer_model: str | None = None,
    tokenizer_file_path: str | None = None,
    max_new_tokens: int = 50,
    context_length: int | None = None,
    temperature: float = 0.0,
    top_k: int | None = None,
    top_p: float | None = None,
    eos_token_id: int | None = None,
    use_cache: bool | None = None,
    device: str = "cpu",
) -> str:
    """Generate and return only the newly generated text, excluding the prompt."""
    if use_cache is None:
        use_cache = family in ("gpt2", "qwen3")
    if use_cache and family not in ("gpt2", "qwen3"):
        raise ValueError(f"{family} does not support the unified KV cache API")
    token_ids, tokenizer = _encode(
        prompt, family, tokenizer_model, tokenizer_file_path, device
    )
    ctx = _context_length(model, context_length)
    from inference.generate_core import generate, generate_qwen3

    generator = generate_qwen3 if family == "qwen3" else generate
    output = generator(
        model,
        token_ids,
        max_new_tokens=max_new_tokens,
        context_length=ctx,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        eos_token_id=eos_token_id,
        use_cache=use_cache,
    )
    new_tokens = output if family == "qwen3" else output[:, token_ids.size(1):]
    return tokenizer.decode(new_tokens.squeeze(0).tolist())


def generate_text_stream(model, prompt: str, family: str = "gpt2", **kwargs):
    """Yield newly decoded token fragments for a single prompt."""
    tokenizer_model = kwargs.pop("tokenizer_model", None)
    tokenizer_file_path = kwargs.pop("tokenizer_file_path", None)
    device = kwargs.pop("device", "cpu")
    max_new_tokens = kwargs.pop("max_new_tokens", 50)
    temperature = kwargs.pop("temperature", 0.0)
    top_k = kwargs.pop("top_k", None)
    top_p = kwargs.pop("top_p", None)
    eos_token_id = kwargs.pop("eos_token_id", None)
    use_cache = kwargs.pop("use_cache", None)
    context_length = kwargs.pop("context_length", None)
    if kwargs:
        raise TypeError(f"Unexpected generation arguments: {', '.join(kwargs)}")
    if use_cache is None:
        use_cache = family in ("gpt2", "qwen3")
    if use_cache and family not in ("gpt2", "qwen3"):
        raise ValueError(f"{family} does not support the unified KV cache API")

    import torch
    from inference.generate_core import sample_next_token
    from layers.kv_cache import KVCache

    token_ids, tokenizer = _encode(
        prompt, family, tokenizer_model, tokenizer_file_path, device
    )
    ctx = _context_length(model, context_length)
    model.eval()
    cache = None
    if use_cache:
        cache = KVCache(n_layers=len(model.trf_blocks))
        model.reset_kv_cache()
        logits = model(token_ids[:, -ctx:], cache=cache)
    else:
        logits = model(token_ids[:, -ctx:])
    with torch.inference_mode():
        for _ in range(max_new_tokens):
            next_idx = sample_next_token(
                logits[:, -1], temperature=temperature, top_k=top_k, top_p=top_p
            )
            if eos_token_id is not None and torch.all(next_idx == eos_token_id):
                break
            yield tokenizer.decode(next_idx.squeeze(0).tolist())
            token_ids = torch.cat((token_ids, next_idx), dim=1)
            if use_cache:
                logits = model(next_idx, cache=cache)
            else:
                logits = model(token_ids[:, -ctx:])
