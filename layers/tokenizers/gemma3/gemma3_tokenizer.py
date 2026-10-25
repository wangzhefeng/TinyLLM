# -*- coding: utf-8 -*-

# ***************************************************
# * File        : gemma3_tokenizer.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2026-10-25
# * Version     : 1.0.102501
# * Description : Gemma3 tokenizer（tokenizers 库加载 tokenizer.json）
# * Link        : link
# * Requirement : huggingface_hub, tokenizers
# ***************************************************

__all__ = ["Gemma3Tokenizer", "ensure_gemma_tokenizer"]


# python libraries
from pathlib import Path

from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer


def ensure_gemma_tokenizer(repo_id: str, local_dir: str) -> Path:
    path = Path(local_dir).expanduser().resolve() / "tokenizer.json"
    if not path.exists():
        downloaded = hf_hub_download(
            repo_id=repo_id,
            filename="tokenizer.json",
            local_dir=str(path.parent),
        )
        path = Path(downloaded)
    if not path.is_file():
        raise FileNotFoundError(f"Unable to provision Gemma tokenizer at {path}")
    return path


class Gemma3Tokenizer:
    def __init__(self, tokenizer_file_path: str):
        tok_file = Path(tokenizer_file_path).expanduser().resolve()
        if not tok_file.is_file():
            raise FileNotFoundError(f"Tokenizer file '{tok_file}' not found")
        self._tok = Tokenizer.from_file(str(tok_file))
        self.n_vocab = self._tok.get_vocab_size()
        self.eos_token_id = self._tok.token_to_id("<eos>")
        self.eot_token_id = self._tok.token_to_id("<end_of_turn>")
        self.pad_token_id = self._tok.token_to_id("<pad>")

    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text).ids

    def decode(self, ids: list[int]) -> str:
        return self._tok.decode(ids, skip_special_tokens=False)
