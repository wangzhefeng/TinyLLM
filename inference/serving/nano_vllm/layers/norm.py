# -*- coding: utf-8 -*-

# ***************************************************
# * File        : norm.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-07-27
# * Version     : 1.1.072816
# * Description : RMSNorm(均方根层归一化)。
# *               支持残差融合：forward(x, residual) 时先相加再归一化，
# *               并返回新的 residual，对应 Qwen3 的 pre-norm 结构。
# * Link        : https://github.com/GeeeekExplorer/nano-vllm
# * Requirement : torch >= 2.0
# ***************************************************

__all__ = [
    "RMSNorm",
]

# python libraries
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


class RMSNorm(nn.Module):
    """
    用于注意力前后的层归一化

    rms_norm(x) = x / sqrt(mean(x^2) + eps) * weight
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        # 在 float32 下计算方差，避免低精度溢出
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dtype)

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            return self._norm(x)
        # 残差融合：先累加得到新残差，再归一化
        residual = x + residual
        return self._norm(residual), residual




# 测试代码 main 函数
def main():
    torch.manual_seed(0)
    norm = RMSNorm(8)
    x = torch.randn(4, 8)
    y = norm(x)
    assert y.shape == x.shape
    # 残差融合
    h, r = norm(x, torch.randn(4, 8))
    assert h.shape == r.shape == x.shape
    print("RMSNorm 测试通过")

if __name__ == "__main__":
    main()
