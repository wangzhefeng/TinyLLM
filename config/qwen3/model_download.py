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
import json
import urllib.request
import warnings
warnings.filterwarnings("ignore")

from utils.log_util import logger

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


def __download_file(url, out_dir=".", backup_url=None):
    """
    Download *url* into *out_dir* with an optional mirror fallback.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = Path(urllib.parse.urlparse(url).path).name
    dest = out_dir / filename

    def _download(u):
        try:
            with urllib.request.urlopen(u) as r:
                size_remote = int(r.headers.get("Content-Length", 0))
                if dest.exists() and dest.stat().st_size == size_remote:
                    print(f"✓ {dest} already up-to-date")
                    return True

                block = 1024 * 1024  # 1 MiB
                downloaded = 0
                with open(dest, "wb") as f:
                    while chunk := r.read(block):
                        f.write(chunk)
                        downloaded += len(chunk)
                        if size_remote:
                            pct = downloaded * 100 // size_remote
                            sys.stdout.write(
                                f"\r{filename}: {pct:3d}% "
                                f"({downloaded // (1024*1024)} MiB / {size_remote // (1024*1024)} MiB)"
                            )
                            sys.stdout.flush()
                if size_remote:
                    sys.stdout.write("\n")
            return True
        except (urllib.error.HTTPError, urllib.error.URLError):
            return False

    if _download(url):
        return dest

    if backup_url:
        print(f"Primary URL ({url}) failed. \nTrying backup URL ({backup_url})...,")
        if _download(backup_url):
            return dest

    raise RuntimeError(f"Failed to download {filename} from both mirrors.")


def download_from_huggingface(kind="base", tokenizer_only=False, repo_id="rasbt/qwen3-from-scratch", revision="main", local_dir="."):
    # model download local dir
    Path(local_dir).mkdir(parents=True, exist_ok=True)
    # model files
    files = {
        "base": {
            "model": "qwen3-0.6B-base.pth", 
            "tokenizer": "tokenizer-base.json"
        },
        "reasoning": {
            "model": "qwen3-0.6B-reasoning.pth", 
            "tokenizer": "tokenizer-reasoning.json"
        },
        "instruction": {
            "model": "",
            "model": "",
        },
    }
    if kind not in files:
        raise ValueError("kind must be 'base' or 'reasoning'")
    
    # model url
    url = "https://huggingface.co/{repo_id}/resolve/{revision}/{filename}"
    backup_root = "https://f001.backblazeb2.com/file/reasoning-from-scratch/qwen3-0.6B"

    # model download
    targets = ["tokenizer"] if tokenizer_only else ["model", "tokenizer"]
    for key in targets:
        # model file name
        filename = files[kind][key]
        # model file url
        primary_url = url.format(repo_id=repo_id, revision=revision, filename=filename)
        backup_url = f"{backup_root}/{filename}"
        # model file out path
        dest_path = os.path.join(local_dir, filename)

        if os.path.exists(dest_path):
            logger.info(f"File already exists: {dest_path}")
        else:
            logger.info(f"Downloading {url} to {dest_path}...")
            __download_file(primary_url, out_dir=local_dir, backup_url=backup_url)
            # urllib.request.urlretrieve(url, dest_path)

    return dest_path


def download_from_huggingface_from_snapshots(repo_id, local_dir):
    from huggingface_hub import hf_hub_download, snapshot_download
    from safetensors.torch import load_file  # or your preferred loader

    # model and tokenizer repo files download
    repo_dir = snapshot_download(repo_id=repo_id, local_dir=local_dir)

    # model and tokenizer path
    model_index_path = os.path.join(repo_dir, "model.safetensors.index.json")
    model_single_file_path = os.path.join(repo_dir, "model.safetensors")
    tokenizer_path = os.path.join(repo_dir, "tokenizer.json")
    
    # model load
    if os.path.exists(model_index_path):
        # Multi-shard model
        with open(model_index_path, "r") as f:
            index = json.load(f)

        weights_dict = {}
        for filename in set(index["weight_map"].values()):
            shard_path = os.path.join(repo_dir, filename)
            shard = load_file(shard_path)
            weights_dict.update(shard)
    elif os.path.exists(model_single_file_path):
        # Single-shard model
        weights_file = hf_hub_download(
            repo_id=repo_id,
            filename="model.safetensors",
            local_dir=local_dir,
        )
        weights_dict = load_file(weights_file)
    else:
        raise FileNotFoundError("No model.safetensors or model.safetensors.index.json found.")

    # tokenizer load
    if not os.path.exists(tokenizer_path):
        tokenizer_file = hf_hub_download(
            repo_id=repo_id,
            filename="tokenizer.json",
            local_dir=local_dir,
        )
        logger.info(f"tokenizer_file: {tokenizer_file}")

    return weights_dict, tokenizer_path




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
