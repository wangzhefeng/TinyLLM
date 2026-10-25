"""JSONL/JSON helpers with atomic writes."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Iterator


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield non-empty JSON objects from a JSONL file."""
    path = Path(path).expanduser()
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(
                    f"{path}:{line_no}: expected a JSON object, got {type(value).__name__}"
                )
            yield value


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return list(iter_jsonl(path))


def read_json(path: str | Path) -> Any:
    with Path(path).expanduser().open("r", encoding="utf-8") as f:
        return json.load(f)


def read_text_label_tsv(path: str | Path) -> tuple[list[str], list[int]]:
    """Read a two-column TSV file (``text<TAB>int-label`` per line); return (texts, labels)."""
    texts: list[str] = []
    labels: list[int] = []
    with Path(path).expanduser().open("r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            text, label = line.rsplit("\t", 1)
            texts.append(text)
            labels.append(int(label))
    return texts, labels


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        tmp_path.write_text(text, encoding="utf-8")
        os.replace(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok=True)


def write_jsonl(
    records: Iterable[dict[str, Any]],
    path: str | Path,
    ensure_ascii: bool = False,
) -> int:
    """Write records as JSONL atomically; return the number of records."""
    path = Path(path).expanduser()
    lines = [
        json.dumps(record, ensure_ascii=ensure_ascii)
        for record in records
    ]
    text = "\n".join(lines)
    if text:
        text += "\n"
    _atomic_write_text(path, text)
    return len(lines)


def write_json(data: Any, path: str | Path, ensure_ascii: bool = False) -> None:
    path = Path(path).expanduser()
    text = json.dumps(data, ensure_ascii=ensure_ascii, indent=2) + "\n"
    _atomic_write_text(path, text)
