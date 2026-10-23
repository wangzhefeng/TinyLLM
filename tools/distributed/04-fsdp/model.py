# -*- coding: utf-8 -*-

# ***************************************************
# * File        : model.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-08-20
# * Version     : 1.0.082014
# * Description : 04 单元的 toy Transformer，统一由
# *               tools/distributed/_common/model.py 提供（供 04/06/07 单元
# *               共用），本文件仅做 re-export 以保持 FSDP 示例的导入结构。
# * Link        : https://github.com/pytorch/examples/tree/main/distributed/FSDP2
# * Requirement : torch >= 2.5
# ***************************************************

import sys
from pathlib import Path

# repo root: tools/distributed/04-fsdp/model.py -> parents[3]
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from tools.distributed._common.model import ModelArgs, Transformer

__all__ = ["ModelArgs", "Transformer"]
