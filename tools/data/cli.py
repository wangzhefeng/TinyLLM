"""Command line entry point for tools.data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .io import iter_jsonl
from .text import dedupe_jsonl, text_fingerprint


def _cmd_dedupe(args: argparse.Namespace) -> int:
    written = dedupe_jsonl(
        args.input,
        args.output,
        key=args.key,
        normalize=not args.no_normalize,
        lowercase=not args.case_sensitive,
        skip_empty=not args.keep_empty,
    )
    print(f"written: {written} -> {args.output}")
    return 0


def _cmd_stats(args: argparse.Namespace) -> int:
    total = 0
    unique: set[str] = set()
    empty = 0
    for record in iter_jsonl(args.input):
        total += 1
        value = record.get(args.key)
        if value is None or (isinstance(value, str) and not value.strip()):
            empty += 1
            continue
        if not isinstance(value, str):
            value = json.dumps(value, ensure_ascii=False, sort_keys=True)
        unique.add(text_fingerprint(value))
    print(
        json.dumps(
            {
                "path": str(Path(args.input)),
                "records": total,
                "unique": len(unique),
                "duplicates": total - empty - len(unique),
                "empty": empty,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tools.data",
        description="JSONL data utilities: stats and exact deduplication.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    dedupe = sub.add_parser("dedupe", help="deduplicate a JSONL file by a text key")
    dedupe.add_argument("input")
    dedupe.add_argument("output")
    dedupe.add_argument("--key", default="text")
    dedupe.add_argument("--no-normalize", action="store_true")
    dedupe.add_argument("--case-sensitive", action="store_true")
    dedupe.add_argument("--keep-empty", action="store_true")
    dedupe.set_defaults(func=_cmd_dedupe)

    stats = sub.add_parser("stats", help="show record/unique/empty counts")
    stats.add_argument("input")
    stats.add_argument("--key", default="text")
    stats.set_defaults(func=_cmd_stats)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)
