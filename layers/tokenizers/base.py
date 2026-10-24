from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class TokenizerProtocol(Protocol):
    @property
    def n_vocab(self) -> int: ...

    @property
    def eos_token_id(self) -> int | None: ...

    @property
    def pad_token_id(self) -> int | None: ...

    def encode(self, text: str, **kwargs) -> list[int]: ...

    def decode(self, token_ids: list[int]) -> str: ...
