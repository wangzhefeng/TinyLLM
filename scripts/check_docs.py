#!/usr/bin/env python3
"""Documentation health check for TinyLLM.

Checks:
1. Markdown pages under docs/ that exceed a line limit.
2. Relative links in docs/ that point to missing files.
3. Markdown files outside docs/ (root README.md and AGENTS.md are allowed).

Usage:
    python scripts/check_docs.py [--max-lines 150]
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
ALLOWED_ROOT_DOCS = {ROOT / "README.md", ROOT / "AGENTS.md"}
LINK_RE = re.compile(r"\]\(([^)]+)\)")
SKIP_DIRS = {".git", ".venv", "__pycache__", "dataset", "downloaded_models", "logs"}


def iter_markdown(base: Path):
    for path in base.rglob("*.md"):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        yield path


def check_size(max_lines: int) -> list[str]:
    issues = []
    for path in iter_markdown(DOCS):
        n = len(path.read_text(encoding="utf-8").splitlines())
        if n > max_lines:
            issues.append(f"[size] {path.relative_to(ROOT)}: {n} lines (> {max_lines})")
    return issues


def check_links() -> list[str]:
    issues = []
    paths = [ROOT / "README.md", ROOT / "AGENTS.md", *iter_markdown(DOCS)]
    for path in paths:
        text = path.read_text(encoding="utf-8")
        # Ignore fenced code blocks: they may contain example markdown links.
        text = re.sub(r"```.*?```", "", text, flags=re.S)
        for raw in LINK_RE.findall(text):
            target = raw.strip()
            if target.startswith(("http://", "https://", "mailto:", "#")):
                continue
            if target.startswith("<") and target.endswith(">"):
                target = target[1:-1]
            target = target.split("#", 1)[0].split("?", 1)[0]
            if not target:
                continue
            target = unquote(target)
            resolved = (path.parent / target).resolve()
            if not resolved.exists():
                issues.append(
                    f"[link] {path.relative_to(ROOT)} -> {target}"
                )
    return issues


def check_outside_docs() -> list[str]:
    issues = []
    for path in iter_markdown(ROOT):
        if DOCS in path.parents:
            continue
        if path in ALLOWED_ROOT_DOCS:
            continue
        issues.append(f"[outside] {path.relative_to(ROOT)}")
    return issues


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-lines", type=int, default=150)
    args = parser.parse_args()

    issues = []
    issues += check_size(args.max_lines)
    issues += check_links()
    issues += check_outside_docs()

    if not issues:
        print("docs check: OK")
        return 0

    print("docs check: issues found\n")
    for issue in issues:
        print(issue)
    print(f"\ntotal: {len(issues)}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
