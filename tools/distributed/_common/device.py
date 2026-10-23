# -*- coding: utf-8 -*-

# ***************************************************
# * File        : device.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-08-20
# * Version     : 1.0.082001
# * Description : 分布式示例的设备选择辅助。分布式场景（gloo/nccl）只支持
# *               CUDA 与 CPU，不使用 MPS（gloo 不支持 MPS）；单进程脚本
# *               可通过 allow_mps=True 使用 MPS。
# * Link        : https://pytorch.org/docs/stable/distributed.html#backends
# * Requirement : torch >= 2.5
# ***************************************************

__all__ = ["get_device", "get_dist_device"]

# python libraries
import torch


def get_device(allow_mps: bool = False) -> torch.device:
    """
    单进程场景的设备选择：cuda > cpu（可选 mps）。

    Args:
        allow_mps: 为 True 且 CUDA 不可用时允许回退到 Apple Silicon MPS。
            分布式训练脚本请勿开启（gloo 后端不支持 MPS 张量通信）。
    """
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if allow_mps and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_dist_device(local_rank: int = 0) -> torch.device:
    """
    分布式场景的设备选择：有 CUDA 时绑定 cuda:{local_rank}，否则回退 CPU。

    每个进程调用一次；CPU 回退使示例可在无 GPU 机器上用 gloo 跑通。
    """
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        return device
    return torch.device("cpu")




# 测试代码 main 函数
def main():
    print(f"get_device(): {get_device()}")
    print(f"get_dist_device(0): {get_dist_device(0)}")

if __name__ == "__main__":
    main()
