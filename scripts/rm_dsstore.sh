#!/usr/bin/env bash
set -euo pipefail

find . -name .DS_Store -print0 | xargs -0 git rm -f --ignore-unmatch
if ! grep -qxF '.DS_Store' .gitignore; then
    printf '\n.DS_Store\n' >> .gitignore
fi
