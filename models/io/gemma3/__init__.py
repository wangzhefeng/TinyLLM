# -*- coding: utf-8 -*-

# ***************************************************
# * File        : __init__.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2026-10-24
# * Version     : 1.0.102421
# * Description : Gemma3 权重下载与加载接口。
# * Requirement : torch / safetensors / huggingface_hub
# ***************************************************

__all__ = ["download_weights", "load_weights_into_gemma"]

from models.io.gemma3.load_weights import download_weights, load_weights_into_gemma  # noqa: F401
