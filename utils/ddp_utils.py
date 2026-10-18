# -*- coding: utf-8 -*-

# ***************************************************
# * File        : ddp.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-08-17
# * Version     : 1.0.081701
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
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP             # model

from utils.log_util import logger

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


def is_dist():
    return dist.is_available() and dist.is_initialized()


# def init_dist():
#     if is_dist():
#         dist.init_process_group(backend="nccl")


# TODO
def init_dist(rank, world_size):
    """
    function to initialize a distributed process group(1 process/GPU)
    this allows communication among processes

    Args:
        rank (int): a unique process ID(Unique identifier of each process)
        world_size (int): total number of processes in the group
    """
    logger.info(f"init dist...")
    # Only set MASTER_ADDR and MASTER_PORT if not already defined by torchrun
    # rank of machine running rank:0 process, assume all GPUs are on the same machine
    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = "localhost"
    logger.info(f"os.environ['MASTER_ADDR']: {os.environ['MASTER_ADDR']}")

    # any free port on the machine
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = "12345"
    logger.info(f"os.environ['MASTER_PORT']: {os.environ['MASTER_PORT']}")
    
    # initialize process group
    # rank = get_global_rank()
    # logger.info(f"rank: {rank}")
    # world_size = get_world_size()
    # logger.info(f"world_size: {world_size}")
    
    if platform.system() == "Windows":
        # Disable libuv because PyTorch for Windows isn't built with support
        os.environ["USE_LIBUV"] = "0"
        # Windows users may have to use "gloo" instead of "nccl" as backend
        # gllo: Facebook Collective Communication Library
        dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    elif torch.cuda.is_available():
        # nccl: NVIDIA Collective Communication Library
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    else:
        # CPU / Apple Silicon (MPS 不支持分布式训练)
        dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    logger.info(f"os.environ['USE_LIBUV']: {os.environ.get('USE_LIBUV')}")
    # set device (only CUDA has one device per process)
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)


def ddp_setup(rank, world_size):
    """
    `torch.multiprocessing.spawn` 方式启动 DDP 时的初始化入口
    """
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "12355")
    init_dist(rank, world_size)


def ddp_setup_custom(rank, world_size):
    """
    自定义 DDP 初始化入口（保留历史脚本兼容）
    """
    ddp_setup(rank, world_size)


def destory_dist():
    logger.info(f"destory dist...")
    # Only set MASTER_ADDR and MASTER_PORT if not already defined by torchrun
    dist.destroy_process_group()


def is_master():
    return get_global_rank() == 0 if is_dist() else True


def get_global_rank():
    """
    全局 Rank: 整个分布式集群中的全局进程编号
    
    torch.distributed.get_rank() —— 全局 Rank（RANK）
        - 定义：在整个分布式训练集群中，所有进程的唯一编号。
        - 范围：从 0 到 world_size - 1，跨机器、跨节点。
        - 来源：由 torch.distributed.init_process_group() 自动分配，基于启动命令（如 torchrun）。
        - 用途：
            - 用于全局通信（如 all_reduce, broadcast）
            - 决定哪个进程保存模型（通常只让 rank == 0 保存）
            - 控制数据采样器（DistributedSampler）如何切分数据
    """
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
    elif torch.distributed.is_initialized():
        rank = dist.get_rank()
    else:
        rank = 0

    return rank


def get_world_size():
    if "WORLD_SIZE" in os.environ:
        world_size = int(os.environ["WORLD_SIZE"])
    elif torch.distributed.is_initialized():
        world_size = dist.get_world_size()
    else:
        world_size = 1
        # world_size = torch.cuda.device_count()

    return world_size


def get_local_rank():
    """
    当前节点（机器）内的进程编号
        - 定义：当前单台机器（节点）内部的进程编号。
        - 范围：从 0 到 (nproc_per_node - 1)，仅限于本机。
        -来源：由 torchrun 或 torch.distributed.launch 自动设置为环境变量：`os.environ["LOCAL_RANK"]`
        - 用途：
            - 绑定当前进程到本机的某一块 GPU（配合 CUDA_VISIBLE_DEVICES）
            - 设置设备：device = torch.device(f'cuda:{local_rank}')
            - 避免多个进程争抢同一块 GPU
    """
    if is_dist() and "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        local_rank = 0

    return local_rank


def dist_gather(o):
    o_all = [None for _ in range(get_world_size())]
    dist.all_gather_object(o_all, o)

    return o_all


def wrap_model(model):
    model = DDP(model, device_ids=[get_global_rank()])

    return model


def print_master_log(rank: int=0):
    if rank == 0:
        # PyTorch version
        logger.info(f"PyTorch version: {torch.__version__}")
        # CUDA version and CUDA capability
        if torch.cuda.is_available():
            # CUDA version
            logger.info(f"CUDA version: {torch.version.cuda}")
            # CUDA capability
            capability = torch.cuda.get_device_capability()
            if capability[0] >= 7:
                torch.set_float32_matmul_precision("high")
                logger.info("Uses tensor cores.")
            else:
                logger.info("Tensor cores not supported on this GPU. Using default precision.")




# 测试代码 main 函数
def main():
    init_dist()
    logger.info(dist.get_rank())

    # logger.info(f"is_dist: {is_dist()}")
    # init_dist()
    # destory_dist()
    # logger.info(f"is_master: {is_master()}")

    # world_size = get_world_size()
    # logger.info(f"world_size: {world_size}")

    # rank = get_global_rank()
    # logger.info(f"rank: {rank}")
    
    # print_master_log()

if __name__ == "__main__":
    main()
