# -*- coding: utf-8 -*-

# ***************************************************
# * File        : __init__.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2026-10-24
# * Version     : 1.0.102421
# * Description : Phi-1.5 权重下载与加载接口。
# * Requirement : torch / safetensors / huggingface_hub
# ***************************************************

__all__ = ["download_phi15", "load_weights_into_phi", "model_with_phi15_weights"]

from models.io.phi.load_phi15_weights import (  # noqa: F401
    download_phi15,
    load_weights_into_phi,
    model_with_phi15_weights,
)
