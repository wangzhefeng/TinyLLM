from typing import List

import torch

from layers.tokenizers.base import TokenizerProtocol
from layers.tokenizers.registry import available_tokenizers, create_tokenizer

__all__ = [
    "TokenizerProtocol",
    "available_tokenizers",
    "choose_tokenizer",
    "text_to_token_ids",
    "token_ids_to_text",
]


def choose_tokenizer(tokenizer_model: str = "tiktoken_gpt2_bpe", **kwargs):
    return create_tokenizer(tokenizer_model, **kwargs)


def text_to_token_ids(text: str, tokenizer_model="tiktoken_gpt2_bpe", **kwargs):
    tokenizer = choose_tokenizer(tokenizer_model, **kwargs) if isinstance(tokenizer_model, str) else tokenizer_model
    encoded = tokenizer.encode(text)
    if any(token_id is None for token_id in encoded):
        raise ValueError("Tokenizer returned an unknown special token id")
    return torch.tensor(encoded, dtype=torch.long).unsqueeze(0)


def token_ids_to_text(token_ids: List, tokenizer_model="tiktoken_gpt2_bpe", **kwargs):
    tokenizer = choose_tokenizer(tokenizer_model, **kwargs) if isinstance(tokenizer_model, str) else tokenizer_model
    return tokenizer.decode(token_ids.squeeze(0).tolist())
