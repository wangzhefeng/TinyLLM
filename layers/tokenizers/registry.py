from __future__ import annotations

from collections.abc import Callable


def _factory(module: str, class_name: str) -> Callable:
    def create(**kwargs):
        from importlib import import_module

        cls = getattr(import_module(module), class_name)
        return cls(**kwargs)

    return create


_TOKENIZERS = {
    "simple_custom": _factory("layers.tokenizers.simple_custom", "SimpleTokenizer"),
    "simple_bpe": _factory("layers.tokenizers.bpe.simple_bpe", "BPETokenizerSimple"),
    # 从零 Unigram：返回未训练的实例，需再调用 train(corpus) 或 load(path)
    "unigram": _factory("layers.tokenizers.unigram.unigram", "UnigramTokenizer"),
    "tiktoken_gpt2_bpe": _factory("layers.tokenizers.bpe.tiktoken_bpe", "TiktokenBPE"),
    "gpt2": _factory("layers.tokenizers.bpe.tiktoken_bpe", "TiktokenBPE"),
    "openai_gpt2_bpe": _factory("layers.tokenizers.bpe.openai_gpt2_bpe", "OpenaiGPT2BPE"),
    "llama2": _factory("layers.tokenizers.sp.llama2_7b_sentencepiece", "Llama27bTokenizer"),
    "llama3-8b": _factory("layers.tokenizers.bpe.llama3_8b_bpe", "Llama38bTokenizer"),
    "qwen3": _factory("layers.tokenizers.qwen3.qwen3_tokenizer", "Qwen3Tokenizer"),
    "gemma3": _factory("layers.tokenizers.gemma", "Gemma3Tokenizer"),
}


def normalize_tokenizer_name(name: str) -> str:
    normalized = name.strip().lower()
    if normalized == "llama3":
        normalized = "llama3-8b"
    if normalized not in _TOKENIZERS:
        available = ", ".join(sorted(_TOKENIZERS))
        raise ValueError(f"Unknown tokenizer '{name}'. Available tokenizers: {available}")
    return normalized


def create_tokenizer(name: str, **kwargs):
    normalized = normalize_tokenizer_name(name)
    try:
        return _TOKENIZERS[normalized](**kwargs)
    except TypeError as error:
        raise TypeError(f"Invalid arguments for tokenizer '{name}': {error}") from error


def available_tokenizers() -> tuple[str, ...]:
    return tuple(sorted(_TOKENIZERS))
