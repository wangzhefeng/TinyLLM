# -*- coding: utf-8 -*-

# ***************************************************
# * File        : extension.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2026-10-18
# * Version     : 1.0.101800
# * Description : tokenizer 扩展（LLMs-from-scratch ch05/09）：
# *               1) 词表对齐 padding（pad_vocab_size）
# *               2) 模型 embedding/输出头扩容（resize_token_embeddings）
# *               3) 特殊 token 扩展（ExtendedTokenizer，tokenizer 无关）
# * Link        : https://github.com/rasbt/LLMs-from-scratch/tree/main/ch05/09_bonus_extending-tokenizers
# * Requirement : torch
# ***************************************************

__all__ = [
    "pad_vocab_size",
    "resize_token_embeddings",
    "ExtendedTokenizer",
]

# python libraries
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)
import re
from typing import Dict, List

import torch
import torch.nn as nn

from utils.log_util import logger

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


def pad_vocab_size(vocab_size: int, multiple: int = 64) -> int:
    """
    把词表大小向上对齐到 multiple 的倍数（提升 matmul 效率，ch05/10）。

    Args:
        vocab_size: 原始词表大小（如 GPT-2 的 50257）
        multiple: 对齐倍数，0 或 1 表示不 padding

    Returns:
        对齐后的词表大小（如 50304）
    """
    if multiple <= 1:
        return vocab_size
    return ((vocab_size + multiple - 1) // multiple) * multiple


def _find_embedding_and_head(model: nn.Module):
    """
    在常见命名中找到 token embedding 与输出头。

    支持：tok_embed/tok_emb（embedding），out_head（输出层）。
    """
    embed = None
    for name in ("tok_embed", "tok_emb"):
        if hasattr(model, name) and isinstance(getattr(model, name), nn.Embedding):
            embed = getattr(model, name)
            break
    head = getattr(model, "out_head", None)
    if embed is None or head is None:
        raise AttributeError(
            "cannot locate token embedding (tok_embed/tok_emb) or output head (out_head) "
            f"on model of type {type(model).__name__}"
        )
    return embed, head


def resize_token_embeddings(model: nn.Module, new_vocab_size: int) -> nn.Module:
    """
    扩容模型的 token embedding 与输出头到 new_vocab_size。

    - 已有权重复制保留，新增行零初始化（不影响原有 token 的表示）；
    - 若 embedding 与输出头权重共享（weight tying），扩容后保持共享。

    Args:
        model: 含 tok_embed/tok_emb 与 out_head 的模型
        new_vocab_size: 新词表大小，必须 >= 原词表大小

    Returns:
        原模型（就地修改）
    """
    embed, head = _find_embedding_and_head(model)
    old_vocab_size, embed_dim = embed.weight.shape
    if new_vocab_size < old_vocab_size:
        raise ValueError(
            f"new_vocab_size ({new_vocab_size}) < current vocab size ({old_vocab_size}); "
            "shrinking is not supported"
        )
    if new_vocab_size == old_vocab_size:
        return model
    tied = head.weight is embed.weight
    device, dtype = embed.weight.device, embed.weight.dtype
    # embedding 扩容
    new_embed = nn.Embedding(new_vocab_size, embed_dim, device=device, dtype=dtype)
    with torch.no_grad():
        new_embed.weight.zero_()
        new_embed.weight[:old_vocab_size] = embed.weight
    # 输出头扩容
    if isinstance(head, nn.Linear):
        new_head = nn.Linear(embed_dim, new_vocab_size, bias=head.bias is not None,
                             device=device, dtype=dtype)
        with torch.no_grad():
            new_head.weight.zero_()
            new_head.weight[:old_vocab_size] = embed.weight if tied else head.weight
            if head.bias is not None:
                new_head.bias.zero_()
                new_head.bias[:old_vocab_size] = head.bias
    else:
        new_head = nn.Embedding(new_vocab_size, embed_dim, device=device, dtype=dtype)
        with torch.no_grad():
            new_head.weight.zero_()
            new_head.weight[:old_vocab_size] = embed.weight
    # 写回模型
    for name in ("tok_embed", "tok_emb"):
        if getattr(model, name, None) is embed:
            setattr(model, name, new_embed)
    if tied:
        new_head.weight = new_embed.weight
    model.out_head = new_head
    logger.info(
        f"Resized token embeddings: {old_vocab_size} -> {new_vocab_size} "
        f"(weight_tied={tied})"
    )

    return model


class ExtendedTokenizer:
    """
    给任意基础 tokenizer 追加特殊 token 的包装器（tokenizer 无关）。

    特殊 token 按字符串精确匹配切分后映射到新增的 id
    （id 从 base tokenizer 的词表大小开始递增），其余文本交给 base tokenizer。

    用法：
        base = choose_tokenizer("tiktoken_gpt2_bpe")
        tok = ExtendedTokenizer(base, {"<|im_start|>": None, "<|im_end|>": None})
        ids = tok.encode("<|im_start|>hello<|im_end|>")
    """

    def __init__(self, base_tokenizer, special_tokens: Dict[str, int]):
        """
        Args:
            base_tokenizer: 需有 encode/decode 方法与 n_vocab 属性（或 n_vocab() 方法）
            special_tokens: {token 字符串: id}；id 传 None 时自动从 n_vocab 开始分配
        """
        self.base_tokenizer = base_tokenizer
        base_n_vocab = self._base_n_vocab(base_tokenizer)
        next_id = base_n_vocab
        self.special_tokens: Dict[str, int] = {}
        for token, token_id in special_tokens.items():
            if token_id is None:
                token_id = next_id
                next_id += 1
            if token_id < base_n_vocab:
                raise ValueError(
                    f"special token id {token_id} for {token!r} conflicts with "
                    f"base vocab (n_vocab={base_n_vocab})"
                )
            self.special_tokens[token] = token_id
            next_id = max(next_id, token_id + 1)
        self._id_to_token = {v: k for k, v in self.special_tokens.items()}
        self._n_vocab = next_id
        # 预编译切分正则（按特殊 token 长度降序，避免前缀冲突）
        self._pattern = None
        if self.special_tokens:
            escaped = sorted((re.escape(t) for t in self.special_tokens), key=len, reverse=True)
            self._pattern = re.compile("(" + "|".join(escaped) + ")")

    @staticmethod
    def _base_n_vocab(base_tokenizer) -> int:
        n_vocab = getattr(base_tokenizer, "n_vocab", None)
        if n_vocab is None:
            raise AttributeError("base tokenizer must expose n_vocab")
        return n_vocab() if callable(n_vocab) else n_vocab

    @property
    def n_vocab(self) -> int:
        return self._n_vocab

    def encode(self, text: str) -> List[int]:
        if self._pattern is None:
            return list(self.base_tokenizer.encode(text))
        token_ids: List[int] = []
        for piece in self._pattern.split(text):
            if not piece:
                continue
            if piece in self.special_tokens:
                token_ids.append(self.special_tokens[piece])
            else:
                token_ids.extend(self.base_tokenizer.encode(piece))
        return token_ids

    def decode(self, tokens: List) -> str:
        parts: List[str] = []
        buffer: List[int] = []

        def flush():
            if buffer:
                parts.append(self.base_tokenizer.decode(list(buffer)))
                buffer.clear()

        for token_id in tokens:
            if token_id in self._id_to_token:
                flush()
                parts.append(self._id_to_token[token_id])
            else:
                buffer.append(token_id)
        flush()
        return "".join(parts)




# 测试代码 main 函数
def main():
    print(pad_vocab_size(50257, 64))   # 50304
    print(pad_vocab_size(50304, 64))   # 50304

if __name__ == "__main__":
    main()
