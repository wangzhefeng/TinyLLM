"""JSONL data utilities: I/O, normalization, deduplication, packing."""

from .io import iter_jsonl, read_json, read_jsonl, write_json, write_jsonl
from .packing import pack_token_ids
from .text import dedupe_jsonl, dedupe_records, normalize_text, text_fingerprint

__all__ = [
    "dedupe_jsonl",
    "dedupe_records",
    "iter_jsonl",
    "normalize_text",
    "pack_token_ids",
    "read_json",
    "read_jsonl",
    "text_fingerprint",
    "write_json",
    "write_jsonl",
]
