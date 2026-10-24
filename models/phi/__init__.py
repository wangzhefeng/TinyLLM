# -*- coding: utf-8 -*-

# ***************************************************
# * File        : __init__.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2026-10-24
# * Version     : 1.0.102421
# * Description : Phi-1.5 教学模型包。
# * Requirement : torch
# ***************************************************

__all__ = ["Model", "PhiAttention", "PhiMLP", "PhiBlock"]

from models.phi.phi15 import Model, PhiAttention, PhiMLP, PhiBlock  # noqa: F401
