"""Token packing helpers for pretraining/SFT batches."""

from __future__ import annotations

from typing import Iterable, Iterator


def pack_token_ids(
    token_ids: Iterable[int],
    max_length: int,
    pad_id: int | None = None,
    drop_last: bool = False,
) -> Iterator[list[int]]:
    """Split a token stream into fixed-length chunks.

    Args:
        token_ids: flat token stream.
        max_length: chunk size (must be > 0).
        pad_id: if set, pad the final short chunk; otherwise yield it as-is.
        drop_last: drop the final short chunk instead of yielding/padding it.
    """
    if max_length <= 0:
        raise ValueError("max_length must be > 0")

    chunk: list[int] = []
    for token in token_ids:
        chunk.append(int(token))
        if len(chunk) == max_length:
            yield chunk
            chunk = []

    if not chunk:
        return
    if drop_last:
        return
    if pad_id is not None:
        chunk.extend([pad_id] * (max_length - len(chunk)))
    yield chunk
