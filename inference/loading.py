# -*- coding: utf-8 -*-

# ***************************************************
# * File        : loading.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2026-10-20
# * Version     : 1.0.102020
# * Description : 统一权重加载入口 load_model()。
# *               按 MODEL_REGISTRY 的 family 分派到 inference/loaders/ 下
# *               各族加载库；重依赖（torch/transformers/huggingface_hub）全部
# *               在函数内懒加载。返回 (model, config, extras)。
# * Requirement   : torch；按 source 需要 transformers / safetensors / tensorflow
# ***************************************************

__all__ = [
    "load_model",
]

# python libraries
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)

from inference.registry import get_model_spec

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


def load_model(model_key: str, source: str = "auto", weights_dir: str | None = None,
               device: str = "cpu"):
    """
    统一模型加载入口。

    Args:
        model_key: MODEL_REGISTRY 中的键，如 "gpt2-124m"、"qwen3-0.6b"
        source: 权重来源。"auto" 取该模型注册的第一个 source；
            gpt2 可选 "hf"(transformers) / "safetensors" / "openai_tf"(官方 TF checkpoint)
        weights_dir: 权重下载/缓存目录，缺省 ./downloaded_models/<family>_model/
        device: 目标设备

    Returns:
        (model, config, extras)：extras 为 dict，qwen3 含 tokenizer_file_path
    """
    spec = get_model_spec(model_key)
    if not spec.available:
        raise NotImplementedError(f"{model_key} 尚未闭环：{spec.note}")
    source = spec.sources[0] if source == "auto" else source
    if source not in spec.sources:
        raise ValueError(f"{model_key} 不支持 source='{source}'，可选：{spec.sources}")
    weights_dir = weights_dir or f"./downloaded_models/{spec.family}_model"

    if spec.family == "gpt2":
        return _load_gpt2(spec, source, weights_dir, device)
    elif spec.family == "qwen3":
        return _load_qwen3(spec, weights_dir, device)
    elif spec.family == "llama":
        return _load_llama(spec, weights_dir, device)
    else:
        raise NotImplementedError(f"family '{spec.family}' 的加载路径未实现")


def _load_gpt2(spec, source: str, weights_dir: str, device: str):
    import torch  # noqa: F401
    from models.gpt2.gpt2_124M import Model
    from models.io.gpt2.load_gpt2_pretrained_weights import model_with_gpt2_weights
    from utils.args_tools import DotDict

    cfgs = DotDict({
        "vocab_size": 50257, "context_length": 1024, "dropout": 0.0,
        "qkv_bias": True, "embed_dim": 768, "n_layers": 12, "n_heads": 12,
        "pretrained_model": spec.config_key, "pretrained_model_path": weights_dir,
    })
    model, config = model_with_gpt2_weights(cfgs, Model, model_source=source)
    model.to(device)
    return model, config, {}


def _load_qwen3(spec, weights_dir: str, device: str):
    from config.qwen3.model_cfgs import get_cfgs
    from models.qwen3.qwen3 import Model
    from models.io.qwen3.model_download import download_from_huggingface_from_snapshots
    from models.io.qwen3.load_weights import load_weights_into_qwen

    config = get_cfgs(spec.config_key)
    local_dir = Path(weights_dir) / Path(spec.repo_id).name
    weights_dict, tokenizer_file_path = download_from_huggingface_from_snapshots(
        repo_id=spec.repo_id, local_dir=str(local_dir),
    )
    model = Model(cfg=config)
    load_weights_into_qwen(model, config, weights_dict)
    del weights_dict
    model.to(device)
    model.eval()
    return model, config, {"tokenizer_file_path": tokenizer_file_path}


def _load_llama(spec, weights_dir: str, device: str):
    from config.llama.model_cfgs import LLAMA2_CONFIG_7B, LLAMA3_CONFIG_8B
    from utils.args_tools import DotDict

    is_llama2 = spec.config_key == "llama2"
    if is_llama2:
        from models.llama.llama2 import Model
        from models.io.llama.meta_llama2_weights_load_hf import (
            download_llama2_model, download_llama2_chat_model, load_weights_into_llama,
        )
        config = DotDict(LLAMA2_CONFIG_7B)
        download = download_llama2_chat_model if spec.chat else download_llama2_model
    else:
        from models.llama.llama3_8B import Model
        from models.io.llama.meta_llama3_weights_load_hf import (
            download_llama3_model, download_llama3_instruct_model, load_weights_into_llama,
        )
        config = DotDict(LLAMA3_CONFIG_8B)
        download = download_llama3_instruct_model if spec.chat else download_llama3_model

    model_dir = Path(weights_dir) / Path(spec.repo_id).name
    weights = download(model_path=str(model_dir))
    model = Model(config)
    load_weights_into_llama(model, weights, config)
    del weights
    model.to(device)
    model.eval()
    return model, config, {}
