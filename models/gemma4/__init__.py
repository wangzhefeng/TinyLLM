# -*- coding: utf-8 -*-

# ***************************************************
# * File        : __init__.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2026-10-24
# * Version     : 1.0.102421
# * Description : Gemma 4（E2B/E4B）教学模型包
# * Link        : https://huggingface.co/google/gemma-4-e2b
# * Requirement : torch
# ***************************************************

__all__ = [
    "Model",
    "layer_pattern",
    "precompute_proportional_rope_params",
]

from models.gemma4.gemma4 import Model, layer_pattern, precompute_proportional_rope_params
