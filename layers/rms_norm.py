# -*- coding: utf-8 -*-

# ***************************************************
# * File        : rms_norm.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-03-21
# * Version     : 1.0.032122
# * Description : description
# * Link        : https://arxiv.org/pdf/1910.07467
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


import torch
import torch.nn as nn

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm) implementation.
    
    Args:
        emb_dim (int): Dimension of the input embeddings
        eps (float, optional): Small value to avoid division by zero. Defaults to 1e-5.
    
    Attributes:
        eps (float): Small value for numerical stability
        emb_dim (int): Dimension of the input embeddings
        weight (nn.Parameter): Learnable scaling parameter
    """
    
    def __init__(self, emb_dim: int, eps: float = 1e-5):
        super().__init__()

        self.eps = eps
        self.emb_dim = emb_dim
        # self.weight = nn.Parameter(torch.ones(emb_dim, dtype=torch.float32))
        self.weight = nn.Parameter(torch.ones(emb_dim)).float()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for RMSNorm.
        
        Args:
            x (torch.Tensor): Input tensor of shape (..., dim)
            
        Returns:
            torch.Tensor: Normalized output tensor
        """
        # Compute mean square with inplace operations for memory efficiency
        means = x.pow(2).mean(dim=-1, keepdim=True)
        # Use fused operation for better numerical stability
        x_normed = x * torch.rsqrt(means.add_(self.eps))

        return x_normed.mul_(self.weight).to(dtype=x.dtype)




# 测试代码 main 函数
def main():
    torch.manual_seed(123)

    # input
    example_batch = torch.randn(2, 3, 4)
    # RMSNorm
    rms_norm = RMSNorm(emb_dim=example_batch.shape[-1], eps=1e-5)
    rms_norm_pytorch = nn.RMSNorm(example_batch.shape[-1], eps=1e-5)
    print(rms_norm(example_batch))
    print(rms_norm_pytorch(example_batch))

    assert torch.allclose(rms_norm(example_batch), rms_norm_pytorch(example_batch)), "RMSNorm test failed"

if __name__ == "__main__":
    main()
