# -*- coding: utf-8 -*-

# ***************************************************
# * File        : nf4.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2026-10-21
# * Version     : 1.0.102121
# * Description : NF4（NormalFloat4）量化原语自研教学实现（QLoRA）。
# *               NF4 = 16 个 [-1,1] 量化点，是按正态分布 4-bit 信息论最优
# *               的分位数（QLoRA 论文附录 A 的标准码本）。
# *               流程：分块 absmax 归一化 -> 最近码本索引（uint8 存储）
# *                     + 每块 scale -> 反量化重建。
# *               注意：QLoRA 完整流程 = 4bit 存储 + bf16 计算 + LoRA 旁路，
# *               本模块只覆盖量化/反量化原语（训练与旁路请组合 layers/lora.py）。
# *               double quantization（对每块 scale 再量化）未实现——留作练习。
# * Link        : QLoRA https://arxiv.org/abs/2305.14314
# * Requirement : torch
# ***************************************************

__all__ = [
    "NF4_CODEBOOK",
    "nf4_quantize",
    "nf4_dequantize",
]

# python libraries
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)

import torch

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]

# QLoRA 论文附录 A 的标准 NF4 码本（16 个分位点，含两个 0 的对称点）
NF4_CODEBOOK = torch.tensor([
    -1.0000000000000000, -0.6961928009986877, -0.5250730514526367,
    -0.3949174880981445, -0.2844413816928863, -0.1847734302282333,
    -0.0910500362515449,  0.0000000000000000,
     0.0795802995562553,  0.1609302014112472,  0.2461123168468475,
     0.3379152417182922,  0.4407098293304443,  0.5626170039176941,
     0.7229568362236023,  1.0000000000000000,
])


def nf4_quantize(x: torch.Tensor, block_size: int = 64):
    """NF4 量化（分块 absmax 归一化 + 最近码本索引）。

    Args:
        x: 任意形状的浮点张量
        block_size: 量化块大小（QLoRA 默认 64）

    Returns:
        indices: uint8 张量（与输入同形状，值为码本下标 0..15）
        scales: fp32 张量（每个 block 一个 absmax scale；全零块的 scale 为 1，
            保证反量化不产生 NaN/除零）
        shape: 输入原形状（元组）
        block_size: 量化块大小
    """
    shape = x.shape
    flat = x.reshape(-1).float()
    n = flat.numel()
    pad = (-n) % block_size
    if pad:
        flat = torch.cat([flat, torch.zeros(pad, device=x.device, dtype=flat.dtype)])
    blocks = flat.reshape(-1, block_size)                       # (num_blocks, block_size)
    # 每块 absmax 归一化到 [-1, 1]
    scales = blocks.abs().amax(dim=1)                           # (num_blocks,)
    scales = torch.where(scales == 0, torch.ones_like(scales), scales)  # 全零块保护
    normed = blocks / scales.unsqueeze(1)                       # (num_blocks, block_size) ∈ [-1, 1]
    # 最近码本索引：对每个元素找 |x - codebook| 最小者
    codebook = NF4_CODEBOOK.to(x.device)                        # (16,)
    dist = (normed.unsqueeze(-1) - codebook).abs()              # (num_blocks, block_size, 16)
    indices = dist.argmin(dim=-1).to(torch.uint8)               # (num_blocks, block_size)
    # 还原输入形状
    indices = indices.reshape(-1)[:n].reshape(shape)
    return indices, scales, shape, block_size


def nf4_dequantize(indices: torch.Tensor, scales: torch.Tensor,
                   shape, block_size: int = 64) -> torch.Tensor:
    """NF4 反量化：按码本还原值并乘回每块 scale。

    Args:
        indices/scales/shape/block_size: nf4_quantize 的返回

    Returns:
        重建的 fp32 张量（与量化前同形状）
    """
    n = 1
    for d in shape:
        n *= d
    flat_idx = indices.reshape(-1)
    pad = (-flat_idx.numel()) % block_size
    if pad:
        flat_idx = torch.cat([flat_idx, torch.zeros(pad, dtype=torch.uint8,
                                                    device=indices.device)])
    blocks_idx = flat_idx.reshape(-1, block_size)
    codebook = NF4_CODEBOOK.to(indices.device)
    values = codebook[blocks_idx.long()]                        # (num_blocks, block_size) ∈ [-1, 1]
    recon = values * scales.to(indices.device).unsqueeze(1)     # 乘回每块 absmax
    return recon.reshape(-1)[:n].reshape(shape)


# 测试代码 main 函数（误差演示；规范测试见 tests/unit/test_nf4.py）
def main():
    import os
    os.environ.setdefault('LOG_NAME', LOGGING_LABEL)
    from utils.log_util import logger

    torch.manual_seed(0)
    x = torch.randn(4, 128) * 3.0                               # N(0, 9) 数据
    idx, scales, shape, bs = nf4_quantize(x)
    x_hat = nf4_dequantize(idx, scales, shape, bs)
    abs_err = (x - x_hat).abs().mean().item()
    rel_err = ((x - x_hat).norm() / x.norm()).item()
    logger.info(f"NF4 roundtrip: abs_mean_err={abs_err:.4f}, rel_err={rel_err:.4f}, "
                f"storage={idx.element_size()}B/elem + {scales.element_size()}B/block64")


if __name__ == "__main__":
    main()
