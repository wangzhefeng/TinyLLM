# -*- coding: utf-8 -*-

# ***************************************************
# * File        : utils.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-07-08
# * Version     : 1.0.070815
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

__all__ = []

# python libraries
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)
import random
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


def seed_torch(seed: int=0):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all()


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def init_dist():
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(dist.get_rank())


def destory_dist():
    dist.destroy_process_group()


def is_dist():
    return dist.is_available() and dist.is_initialized()


def is_master():
    return dist.get_rank() == 0 if is_dist() else True


def get_world_size():
    return dist.get_world_size() if is_dist() else 1


def get_rank():
    return dist.get_rank() if is_dist() else 0


def dist_gather(o):
    o_all = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(o_all, o)

    return o_all


def wrap_model(model):
    return DDP(model, device_ids=[dist.get_rank()])




# 测试代码 main 函数
def main():
    print(torch.initial_seed())
    print(torch.initial_seed() % 2**32)

if __name__ == "__main__":
    main()
