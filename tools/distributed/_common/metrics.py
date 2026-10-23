# -*- coding: utf-8 -*-

# ***************************************************
# * File        : metrics.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-08-20
# * Version     : 1.0.082005
# * Description : 分布式示例统一的指标输出辅助：CUDA 峰值显存查询（CPU 返回
# *               None）与格式统一的 step 日志，使各单元的吞吐/显存输出可
# *               横向对比。日志仅在 rank 0 打印。
# * Link        : https://pytorch.org/docs/stable/cuda.html#torch.cuda.max_memory_allocated
# * Requirement : torch >= 2.5
# ***************************************************

__all__ = ["peak_memory_mb", "log_step_metrics"]

# python libraries
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

import torch

from tools.distributed._common.dist import is_main_process


def peak_memory_mb(device=None) -> float | None:
    """
    返回当前 CUDA 设备的峰值显存占用（MB）。CPU / MPS 无对应计数器，
    返回 None（调用方应打印 "n/a" 而非报错）。
    """
    if not torch.cuda.is_available():
        return None
    return torch.cuda.max_memory_allocated(device) / (1024 ** 2)


def log_step_metrics(
    step: int,
    loss: float,
    tokens_per_sec: float | None = None,
    extra: str = "",
) -> None:
    """
    仅 rank 0 打印格式统一的 step 指标行：

        step  10 | loss 0.6931 | 12345 tok/s | mem 1024.0MB | <extra>
    """
    if not is_main_process():
        return
    parts = [f"step {step:>5d} | loss {loss:.4f}"]
    if tokens_per_sec is not None:
        parts.append(f"{tokens_per_sec:.0f} tok/s")
    mem = peak_memory_mb()
    parts.append(f"mem {mem:.1f}MB" if mem is not None else "mem n/a")
    if extra:
        parts.append(extra)
    print(" | ".join(parts))




# 测试代码 main 函数
def main():
    log_step_metrics(step=0, loss=0.693, tokens_per_sec=1000.0, extra="demo")
    print(f"peak_memory_mb(): {peak_memory_mb()}")

if __name__ == "__main__":
    main()
