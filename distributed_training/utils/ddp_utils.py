# -*- coding: utf-8 -*-

# ***************************************************
# * File        : utils.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-07-03
# * Version     : 1.0.070317
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
import platform
import warnings
warnings.filterwarnings("ignore")

import torch
from torch.utils.data.distributed import DistributedSampler              # data
from torch.nn.parallel import DistributedDataParallel as DDP             # model
from torch.distributed import init_process_group, destroy_process_group  # process
import torch.multiprocessing as mp                                       # device

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


def ddp_setup_custom(rank: int, world_size: int):
    """
    function to initialize a distributed process group(1 process/GPU)
    this allows communication among processes

    Args:
        rank (int): a unique process ID(Unique identifier of each process)
        world_size (int): total number of processes in the group
    """
    # Only set MASTER_ADDR and MASTER_PORT if not already defined by torchrun
    # rank of machine running rank:0 process, assume all GPUs are on the same machine
    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = "localhost"
    # any free port on the machine
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = "12345"
    
    # initialize process group
    if platform.system() == "Windows":
        # Disable libuv because PyTorch for Windows isn't built with support
        os.environ["USE_LIBUV"] = "0"
        # Windows users may have to use "gloo" instead of "nccl" as backend
        # gllo: Facebook Collective Communication Library
        init_process_group(backend="gloo", rank=rank, world_size=world_size)
    else:
        # nccl: NVIDIA Collective Communication Library
        init_process_group(backend="nccl", rank=rank, world_size=world_size)
    # set deivce
    torch.cuda.set_device(rank)


def ddp_setup():
    init_process_group(backend="gloo|nccl")
    torch.cuda.set_device(int(os.environ["LOCAL-RANK"]))




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
