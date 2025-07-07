# -*- coding: utf-8 -*-

# ***************************************************
# * File        : main.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-07-03
# * Version     : 1.0.070315
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

__all__ = []

# python libraries
import os
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)
from typing import Dict
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from torch.distributed.fsdp import (
    FullyShardedDataParallel,
    CPUOffload,
)
from torch.distributed.fsdp.wrap import (
    enable_wrap,
    wrap
)

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]
os.environ['LOG_NAME'] = LOGGING_LABEL
from utils.log_util import logger


# model
model = None

# fsdp
wrapper_kwargs = Dict(
    cup_offload = CPUOffload(offload_prams=True)
)
with enable_wrap(wrapper_cls=FullyShardedDataParallel, **wrapper_kwargs):
    fsdp_model = wrap(model)

# optimizer
optimizer = torch.optim.Adam(fsdp_model.parameters(), lr=1e-4)

# criterion
criterion = nn.CrossEntropyLoss()

# training
for sample, label in next_batch():
    out = fsdp_model(sample)
    loss = criterion(out, label)
    loss.backward()
    optimizer.step()




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
