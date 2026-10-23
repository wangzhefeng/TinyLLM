# -*- coding: utf-8 -*-

# ***************************************************
# * File        : inspect_utils.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-08-20
# * Version     : 1.0.082016
# * Description : 04 单元的模型检查辅助：验证 fully_shard 后参数被切分为
# *               DTensor(Shard(0))、验证混合精度下参数 dtype 为 bfloat16。
# *               （上游名为 utils.py，改名以避免遮蔽仓库根的 utils/ 包。）
# * Link        : https://github.com/pytorch/examples/blob/main/distributed/FSDP2/utils.py
# * Requirement : torch >= 2.5
# ***************************************************

import torch
from model import Transformer
try:
    from torch.distributed.fsdp import FSDPModule
except ImportError:
    from torch.distributed._composable.fsdp import FSDPModule
from torch.distributed.tensor import Shard


def inspect_model(model: FSDPModule):
    assert isinstance(model, Transformer)
    assert isinstance(model, FSDPModule)

    if torch.distributed.get_rank() == 0:
        print(model)

    for param in model.parameters():
        assert param.placements == (Shard(0),)
        assert param.dtype == torch.float32
        # print(param.get_local_tensor())


def inspect_mixed_precision(model: FSDPModule):
    model.unshard()
    for param in model.parameters(recurse=False):
        assert param.dtype == torch.bfloat16
    model.reshard()
