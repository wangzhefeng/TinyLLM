"""Text normalization and exact deduplication helpers."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable, Iterator

from .io import iter_jsonl, write_jsonl

_WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(
    text: str,
    lowercase: bool = False,
    collapse_whitespace: bool = True,
) -> str:
    """Normalize text for exact-match deduplication."""
    if collapse_whitespace:
        text = _WHITESPACE_RE.sub(" ", text).strip()
    if lowercase:
        text = text.lower()
    return text


def text_fingerprint(
    text: str,
    normalize: bool = True,
    lowercase: bool = True,
) -> str:
    """Return a SHA-256 fingerprint for a text value."""
    if normalize:
        text = normalize_text(text, lowercase=lowercase)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def dedupe_records(
    records: Iterable[dict[str, Any]],
    key: str = "text",
    normalize: bool = True,
    lowercase: bool = True,
    skip_empty: bool = True,
) -> Iterator[dict[str, Any]]:
    """Yield records with duplicate values at ``key`` removed, preserving order."""
    seen: set[str] = set()
    for record in records:
        value = record.get(key)
        if value is None:
            if skip_empty:
                continue
            value = ""
        if not isinstance(value, str):
            value = json.dumps(value, ensure_ascii=False, sort_keys=True)
        if skip_empty and not value.strip():
            continue
        fingerprint = text_fingerprint(value, normalize=normalize, lowercase=lowercase)
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        yield record


def dedupe_jsonl(
    input_path: str | Path,
    output_path: str | Path,
    key: str = "text",
    normalize: bool = True,
    lowercase: bool = True,
    skip_empty: bool = True,
) -> int:
    """Deduplicate a JSONL file and return the number of written records."""
    records = dedupe_records(
        iter_jsonl(input_path),
        key=key,
        normalize=normalize,
        lowercase=lowercase,
        skip_empty=skip_empty,
    )
    return write_jsonl(records, output_path)
