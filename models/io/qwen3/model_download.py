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
from pathlib import Path
import json
import urllib.request

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


def download_from_huggingface(kind="base", tokenizer_only=False, repo_id="rasbt/qwen3-from-scratch", revision="main", local_dir=".", variant=None, step=None):
    """按 kind 下载模型/权重文件，返回 (llm_model_path, tokenizer_path)。

    kind 说明：
      - "base" / "reasoning"：rasbt/qwen3-from-scratch 的 base/reasoning
        .pth + tokenizer（rasbt/qwen3-from-scratch 仓，backblaze 备份）
      - "grpo-checkpoint"：ch07 GRPO 训练 checkpoint，
        repo rasbt/qwen3-from-scratch-grpo-checkpoints（variant:
        tracking / clip_ratio / kl / format_reward；默认 clip_ratio，
        step 默认 00050）。README: ch07/04_download_trainining_checkpoints
        （https://raw.githubusercontent.com/rasbt/reasoning-from-scratch/main/ch07/04_download_trainining_checkpoints/README.md）。
        tokenizer：tracking/clip_ratio/kl 用 base，format_reward 用 reasoning
        （2026-10-24 已按 README 与 HF 仓库文件列表核对）
      - "distill-checkpoint"：ch08 蒸馏训练 checkpoint，
        repo rasbt/qwen3-from-scratch-distill-checkpoints（variant:
        deepseek_r1 / qwen3_235b_a22b；默认 deepseek_r1，step 默认 06682）。
        README: ch08/05_download_training_checkpoints
        （https://raw.githubusercontent.com/rasbt/reasoning-from-scratch/main/ch08/05_download_training_checkpoints/README.md）。
        tokenizer 一律用 reasoning（2026-10-24 已核对）

    checkpoint kind 的 step 必须在该 variant 的已保存步骤列表内。
    """
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
    }
    checkpoint_specs = {
        "grpo-checkpoint": {
            "repo_id": "rasbt/qwen3-from-scratch-grpo-checkpoints",
            "default_variant": "clip_ratio",
            "default_step": "00050",
            "checkpoint_file": "{folder}/checkpoints/qwen3-0.6B-rlvr-grpo-step{step}.pth",
            # README：format_reward 自 reasoning 模型训练而来，配 reasoning tokenizer
            "variants": {
                "tracking": {"folder": "7_3_plus_tracking"},
                "clip_ratio": {"folder": "7_4_plus_clip_ratio"},
                "kl": {"folder": "7_5_plus_kl"},
                "format_reward": {
                    "folder": "7_6_plus_format_reward",
                    "tokenizer": "tokenizer-reasoning.json",
                },
            },
        },
        "distill-checkpoint": {
            "repo_id": "rasbt/qwen3-from-scratch-distill-checkpoints",
            "default_variant": "deepseek_r1",
            "default_step": "06682",
            "tokenizer": "tokenizer-reasoning.json",
            "checkpoint_file": "{folder}/checkpoints/qwen3-0.6B-distill-step{step}-epoch{epoch}.pth",
            "variants": {
                "deepseek_r1": {
                    "folder": "ch08_distill_deepseek_r1",
                    "steps": ["06682", "13364", "20046"],
                },
                "qwen3_235b_a22b": {
                    "folder": "ch08_distill_qwen3_235b_a22b",
                    "steps": ["05746", "11492", "17238"],
                },
            },
        },
    }
    if kind not in files and kind not in checkpoint_specs:
        raise ValueError(
            "kind must be 'base', 'reasoning', 'grpo-checkpoint' or 'distill-checkpoint'"
        )

    # model url
    url = "https://huggingface.co/{repo_id}/resolve/{revision}/{filename}"
    backup_root = "https://f001.backblazeb2.com/file/reasoning-from-scratch/qwen3-0.6B"

    if kind in checkpoint_specs:
        return __download_checkpoint(
            checkpoint_specs[kind], tokenizer_only=tokenizer_only,
            repo_id=repo_id, revision=revision, local_dir=local_dir,
            url_template=url, backup_root=backup_root,
            variant=variant, step=step,
        )

    # model download
    llm_model_path = None
    tokenizer_path = None
    targets = ["tokenizer"] if tokenizer_only else ["model", "tokenizer"]
    for key in targets:
        # model file name
        filename = files[kind][key]
        # model file url
        primary_url = url.format(repo_id=repo_id, revision=revision, filename=filename)
        backup_url = f"{backup_root}/{filename}"
        # model file out path
        dest_path = os.path.join(local_dir, filename)
        # model file download
        if os.path.exists(dest_path):
            logger.info(f"File already exists: {dest_path}")
        else:
            logger.info(f"Downloading {url} to {dest_path}...")
            __download_file(primary_url, out_dir=local_dir, backup_url=backup_url)
            # urllib.request.urlretrieve(url, dest_path)
        # return model and tokenizer file path
        if key == "model":
            llm_model_path = dest_path
        elif key == "tokenizer":
            tokenizer_path = dest_path

    return llm_model_path, tokenizer_path


def __download_checkpoint(spec, tokenizer_only=False, repo_id="rasbt/qwen3-from-scratch",
                          revision="main", local_dir=".", url_template=None,
                          backup_root=None, variant=None, step=None):
    """下载训练 checkpoint（grpo/distill kind）：checkpoint 文件 + 配套 tokenizer。"""
    variant = variant or spec["default_variant"]
    if variant not in spec["variants"]:
        raise ValueError(
            f"variant must be one of {sorted(spec['variants'])} for {spec['repo_id']}"
        )
    info = dict(spec["variants"][variant])
    tokenizer_file = info.get("tokenizer", spec.get("tokenizer"))
    if "steps" in info:
        step = str(step or spec["default_step"])
        if step not in info["steps"]:
            raise ValueError(
                f"step must be one of {info['steps']} for variant '{variant}'"
            )
        epoch = info["steps"].index(step) + 1
        filename = spec["checkpoint_file"].format(
            folder=info["folder"], step=step, epoch=epoch,
        )
    else:
        step = str(step or spec["default_step"]).zfill(5)
        filename = spec["checkpoint_file"].format(folder=info["folder"], step=step)

    # tokenizer url（主仓 + backblaze 备份）
    tokenizer_url = url_template.format(
        repo_id=repo_id, revision=revision, filename=tokenizer_file,
    )
    tokenizer_dest = os.path.join(local_dir, tokenizer_file)
    if not os.path.exists(tokenizer_dest):
        __download_file(
            tokenizer_url, out_dir=local_dir,
            backup_url=f"{backup_root}/{tokenizer_file}",
        )
    if tokenizer_only:
        return None, tokenizer_dest

    # checkpoint url（checkpoint 仓无 backblaze 备份）
    checkpoint_url = url_template.format(
        repo_id=spec["repo_id"], revision=revision, filename=filename,
    )
    dest_path = os.path.join(local_dir, Path(filename).name)
    if not os.path.exists(dest_path):
        logger.info(f"Downloading {checkpoint_url} to {dest_path}...")
        __download_file(checkpoint_url, out_dir=local_dir)

    return dest_path, tokenizer_dest


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
