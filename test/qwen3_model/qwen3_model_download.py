# -*- coding: utf-8 -*-

# ***************************************************
# * File        : qwen3_model_download.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-10-02
# * Version     : 1.0.100217
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

__all__ = []

# python libraries
import os
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)
import warnings
warnings.filterwarnings("ignore")

from utils.llm.download_file import download_file

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


def download_qwen3_small(kind="base", tokenizer_only=False, out_dir="."):
    files = {
        "base": {
            "model": "qwen3-0.6B-base.pth", 
            "tokenizer": "tokenizer-base.json"
        },
        "reasoning": {
            "model": "qwen3-0.6B-reasoning.pth", 
            "tokenizer": "tokenizer-reasoning.json"
        },
    }
    if kind not in files:
        raise ValueError("kind must be 'base' or 'reasoning'")

    repo = "rasbt/qwen3-from-scratch"
    hf_fmt = "https://huggingface.co/{repo}/resolve/main/{file}"
    backup_root = "https://f001.backblazeb2.com/file/reasoning-from-scratch/qwen3-0.6B"
    targets = ["tokenizer"] if tokenizer_only else ["model", "tokenizer"]

    for key in targets:
        fname = files[kind][key]
        primary = hf_fmt.format(repo=repo, file=fname)
        backup = f"{backup_root}/{fname}"
        download_file(primary, out_dir=out_dir, backup_url=backup)




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
