# -*- coding: utf-8 -*-

# ***************************************************
# * File        : __init__.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2026-10-25
# * Version     : 1.0.102511
# * Description : inference 公共 API（懒加载，import 本包无副作用）：
# *                 from inference import load_model, generate_text
# *               模型注册表见 models.registry（MODEL_REGISTRY /
# *               ModelSpec / list_models / get_model_spec）。
# *               注意：serving/nano_vllm 为独立子包，不在此导出
# *               （其内部以脚本方式运行，见 docs/30-inference/nano-vllm.md）。
# * Requirement   : 无重依赖（懒加载）
# ***************************************************

__all__ = [
    "load_model",
    "generate_text",
    "generate_text_stream",
]


def __getattr__(name):
    """PEP 562 懒加载：避免 import inference 时拉起 torch/transformers。"""
    if name == "load_model":
        from inference.loading import load_model
        return load_model
    if name in ("generate_text", "generate_text_stream"):
        from inference import generation
        return getattr(generation, name)
    raise AttributeError(f"module 'inference' has no attribute {name!r}")
