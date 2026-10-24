# -*- coding: utf-8 -*-

# ***************************************************
# * File        : dist.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-08-20
# * Version     : 1.0.082004
# * Description : 分布式示例共用的进程组启动辅助。进程组初始化、rank 查询
# *               由 tools 独立实现，不依赖主链路 utils；包含：
# *               torchrun 环境变量入口、mp.spawn 入口、rank0 打印、
# *               统一 checkpoint 目录。脚本从任意 cwd 运行均可（仓库根
# *               基于 __file__ 计算并加入 sys.path）。
# * Link        : https://pytorch.org/docs/stable/distributed.html
# * Requirement : torch >= 2.5；gloo（CPU）或 nccl（CUDA）后端
# ***************************************************

__all__ = [
    "REPO_ROOT",
    "setup",
    "setup_spawn",
    "cleanup",
    "is_dist",
    "is_main_process",
    "rank0_print",
    "get_rank",
    "get_world_size",
    "get_local_rank",
    "get_checkpoint_dir",
]

# python libraries
import os
import sys
from pathlib import Path

# repo root: tools/distributed/_common/dist.py -> parents[3]
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

import torch
import torch.distributed as dist

def is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_global_rank() -> int:
    return dist.get_rank() if is_dist() else int(os.environ.get("RANK", 0))


def get_world_size() -> int:
    return dist.get_world_size() if is_dist() else int(os.environ.get("WORLD_SIZE", 1))


def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", 0))


def get_rank() -> int:
    """全局 rank（未初始化时返回 0）。"""
    return get_global_rank()


def _backend() -> str:
    return "nccl" if torch.cuda.is_available() else "gloo"


def setup(timeout=None) -> tuple[int, int, int]:
    """
    torchrun / torch.distributed.run 场景的初始化入口：
    从环境变量读取 RANK / WORLD_SIZE / LOCAL_RANK，backend 按 CUDA 可用性
    自动选择 nccl/gloo，并把 CUDA 设备绑定到 LOCAL_RANK。

    Returns:
        (rank, world_size, local_rank)

    重复调用安全（已初始化则直接返回）。
    """
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if not is_dist():
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        dist.init_process_group(backend=_backend(), rank=rank, world_size=world_size, timeout=timeout)
    return rank, world_size, local_rank


def setup_spawn(rank: int, world_size: int) -> int:
    """
    torch.multiprocessing.spawn 场景的初始化入口（mp.spawn 会把 rank 作为
    第一个参数传入子进程）。设置 MASTER_ADDR/MASTER_PORT 默认值后初始化
    进程组，返回 local_rank（spawn 单机场景下等于 rank）。
    """
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "12355")
    if not is_dist():
        if torch.cuda.is_available():
            torch.cuda.set_device(rank)
        dist.init_process_group(backend=_backend(), rank=rank, world_size=world_size)
    return rank


def cleanup() -> None:
    """销毁进程组（未初始化时为 no-op）。"""
    if is_dist():
        dist.destroy_process_group()


def is_main_process() -> bool:
    """是否为全局 rank 0（未初始化时视为 True）。"""
    return get_rank() == 0


def rank0_print(*args, **kwargs) -> None:
    """仅在 rank 0 打印，用于分布式脚本的统一日志输出。"""
    if is_main_process():
        print(*args, **kwargs)


def get_checkpoint_dir(unit: str) -> Path:
    """
    统一的 checkpoint 输出目录：<repo>/results/checkpoints/distributed/<unit>/。
    基于 __file__ 计算仓库根，与运行时的 cwd 无关。
    """
    path = REPO_ROOT / "results" / "checkpoints" / "distributed" / unit
    path.mkdir(parents=True, exist_ok=True)
    return path




# 测试代码 main 函数
def main():
    print(f"REPO_ROOT: {REPO_ROOT}")
    print(f"backend: {_backend()}")
    print(f"is_main_process: {is_main_process()}")
    print(f"checkpoint dir: {get_checkpoint_dir('test')}")

if __name__ == "__main__":
    main()
