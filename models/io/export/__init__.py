# -*- coding: utf-8 -*-

# ***************************************************
# * File        : __init__.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2026-10-24
# * Version     : 1.0.102421
# * Description : 本地权重 -> Hugging Face Transformers 导出接口
# *               （Qwen3 命名；TinyLLMForCausalLM 懒加载包装器）。
# * Requirement : torch / safetensors；transformers 仅 wrapper 需要
# ***************************************************

__all__ = ["export_to_hf", "TinyLLMConfig", "TinyLLMForCausalLM"]

from models.io.export.hf_export import export_to_hf  # noqa: F401


def __getattr__(name):
    # TinyLLMConfig / TinyLLMForCausalLM 由 hf_export 按需懒加载（transformers 依赖）
    if name in ("TinyLLMConfig", "TinyLLMForCausalLM"):
        return getattr(__import__(
            "models.io.export.hf_export", fromlist=["hf_export"]
        ), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
