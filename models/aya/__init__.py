# -*- coding: utf-8 -*-

# ***************************************************
# * File        : __init__.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2026-10-24
# * Version     : 1.0.102421
# * Description : Tiny Aya（Cohere，260M）教学模型包。
# *               注意：架构未经官方 config 核实（HF 仓库门控 401），
# *               详见 models/aya/aya.py 头部说明。
# * Link        : https://huggingface.co/cohere-for-ai/tinyaya
# * Requirement : torch
# ***************************************************

__all__ = [
    "ComplexAffinePositionEncoding",
    "GroupedQueryAttentionAya",
    "TransformerBlockAya",
    "Model",
]

from models.aya.aya import (
    ComplexAffinePositionEncoding,
    GroupedQueryAttentionAya,
    TransformerBlockAya,
    Model,
)
