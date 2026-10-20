# -*- coding: utf-8 -*-

# ***************************************************
# * File        : registry.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2026-10-20
# * Version     : 1.0.102020
# * Description : MODEL_REGISTRY：通用推断模块的模型注册表。
# *               每个条目声明模型族（family）、配置来源、权重来源（sources）、
# *               默认 tokenizer 与 eos，供 inference.loading.load_model 统一加载。
# *               本模块只放静态元数据，import 时无任何副作用（不下载、不 import torch）。
# * Requirement   : 无重依赖
# ***************************************************

__all__ = [
    "ModelSpec",
    "MODEL_REGISTRY",
    "list_models",
    "get_model_spec",
]

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelSpec:
    """
    模型注册条目。

    Args:
        family: 模型族（gpt2 / qwen3 / llama / gemma3），决定加载与生成路径
        config_key: 传给各族 config 助手的键（如 "0.6B"、"gpt2-small (124M)"）
        repo_id: 默认权重仓库（HF repo id 或 OpenAI 模型尺寸）
        sources: 可用的权重来源（"auto" 时取第一个）
        tokenizer_model: layers.tokenizers 的 tokenizer 名称（None 表示随权重目录自带）
        eos_token_id: 默认 eos；tokenizer 可用时以 tokenizer 为准
        chat: 是否为对话/指令变体
        available: False 表示加载路径尚未闭环（WIP）
        note: 备注
    """
    family: str
    config_key: str
    repo_id: str
    sources: tuple = ("hf",)
    tokenizer_model: str | None = None
    eos_token_id: int | None = None
    chat: bool = False
    available: bool = True
    note: str = ""


MODEL_REGISTRY = {
    # ---------------- GPT-2（OpenAI 官方权重，三种来源） ----------------
    "gpt2-124m": ModelSpec(
        family="gpt2", config_key="gpt2-small (124M)", repo_id="124M",
        sources=("hf", "safetensors", "openai_tf"),
        tokenizer_model="tiktoken_gpt2_bpe", eos_token_id=50256,
    ),
    "gpt2-355m": ModelSpec(
        family="gpt2", config_key="gpt2-medium (355M)", repo_id="355M",
        sources=("hf", "safetensors", "openai_tf"),
        tokenizer_model="tiktoken_gpt2_bpe", eos_token_id=50256,
    ),
    "gpt2-774m": ModelSpec(
        family="gpt2", config_key="gpt2-large (774M)", repo_id="774M",
        sources=("hf", "safetensors", "openai_tf"),
        tokenizer_model="tiktoken_gpt2_bpe", eos_token_id=50256,
    ),
    "gpt2-1558m": ModelSpec(
        family="gpt2", config_key="gpt2-xl (1558M)", repo_id="1558M",
        sources=("hf", "safetensors", "openai_tf"),
        tokenizer_model="tiktoken_gpt2_bpe", eos_token_id=50256,
    ),
    # ---------------- Qwen3（HF snapshot 下载） ----------------
    "qwen3-0.6b": ModelSpec(
        family="qwen3", config_key="0.6B", repo_id="Qwen/Qwen3-0.6B",
        sources=("hf_snapshot",), chat=True,
        note="reasoning/instruct 混合模型；tokenizer.json 随仓库下载",
    ),
    "qwen3-0.6b-base": ModelSpec(
        family="qwen3", config_key="0.6B", repo_id="Qwen/Qwen3-0.6B-Base",
        sources=("hf_snapshot",),
    ),
    "qwen3-30b-a3b": ModelSpec(
        family="qwen3", config_key="30B-A3B", repo_id="Qwen/Qwen3-30B-A3B",
        sources=("hf_snapshot",), chat=True, note="MoE",
    ),
    # ---------------- Llama（HF gated，需要授权 token） ----------------
    "llama2-7b": ModelSpec(
        family="llama", config_key="llama2", repo_id="meta-llama/Llama-2-7b",
        sources=("hf_gated",), tokenizer_model="llama2", eos_token_id=2,
    ),
    "llama2-7b-chat": ModelSpec(
        family="llama", config_key="llama2", repo_id="meta-llama/Llama-2-7b-chat",
        sources=("hf_gated",), tokenizer_model="llama2", eos_token_id=2, chat=True,
    ),
    "llama3-8b": ModelSpec(
        family="llama", config_key="llama3", repo_id="meta-llama/Meta-Llama-3-8B",
        sources=("hf_gated",), tokenizer_model="llama3-8B", eos_token_id=128001,
    ),
    "llama3-8b-instruct": ModelSpec(
        family="llama", config_key="llama3", repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
        sources=("hf_gated",), tokenizer_model="llama3-8B", eos_token_id=128009, chat=True,
    ),
    # ---------------- Gemma3（WIP：模型组装未闭环） ----------------
    "gemma3-270m": ModelSpec(
        family="gemma3", config_key="270m", repo_id="google/gemma-3-270m",
        sources=("hf_snapshot",), available=False,
        note="WIP：loaders/gemma3 只有下载与加载函数，模型组装未接入",
    ),
}


def list_models(available_only: bool = False) -> list:
    """列出注册的模型 key。"""
    keys = sorted(MODEL_REGISTRY.keys())
    if available_only:
        keys = [k for k in keys if MODEL_REGISTRY[k].available]
    return keys


def get_model_spec(model_key: str) -> ModelSpec:
    """按 key 取注册条目，未知 key 给出可用列表报错。"""
    try:
        return MODEL_REGISTRY[model_key]
    except KeyError:
        raise KeyError(
            f"未知模型 '{model_key}'。可用：{', '.join(list_models())}"
        ) from None
