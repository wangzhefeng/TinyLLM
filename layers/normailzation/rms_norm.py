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
        embed_dim (int): Dimension of the input embeddings
        eps (float, optional): Small value to avoid division by zero. Defaults to 1e-5.
    
    Attributes:
        eps (float): Small value for numerical stability
        embed_dim (int): Dimension of the input embeddings
        weight (nn.Parameter): Learnable scaling parameter
    """
    def __init__(self, embed_dim: int, eps: float = 1e-5):
        super().__init__()

        self.eps = eps
        self.embed_dim = embed_dim
        # self.weight = nn.Parameter(torch.ones(embed_dim, dtype=torch.float32))
        self.weight = nn.Parameter(torch.ones(embed_dim)).float()
    
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


class RMSNorm_Qwen(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm) implementation.
    
    Args:
        embed_dim (int): Dimension of the input embeddings
        eps (float, optional): Small value to avoid division by zero. Defaults to 1e-5.
    
    Attributes:
        eps (float): Small value for numerical stability
        embed_dim (int): Dimension of the input embeddings
        weight (nn.Parameter): Learnable scaling parameter
    """
    def __init__(self, embed_dim: int, eps: float = 1e-6, bias: bool=False, compatible: bool=True):
        super().__init__()

        self.eps = eps
        self.compatible = compatible
        self.scale = nn.Parameter(torch.ones(embed_dim))
        self.shift = nn.Parameter(torch.zeros(embed_dim)) if bias else None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for RMSNorm.
        
        Args:
            x (torch.Tensor): Input tensor of shape (..., dim)
            
        Returns:
            torch.Tensor: Normalized output tensor
        """
        # input dtype
        input_dtype = x.dtype
        # compatible
        if self.compatible:
            x = x.to(torch.float32)
        # Compute mean square with inplace operations for memory efficiency
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        # Use fused operation for better numerical stability
        norm_x = x * torch.rsqrt(variance + self.eps)
        norm_x = norm_x * self.scale

        if self.shift is not None:
            norm_x = norm_x + self.shift

        return norm_x.to(input_dtype)


class RMSNorm_Gemma3(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm) implementation.
    
    Args:
        embed_dim (int): Dimension of the input embeddings
        eps (float, optional): Small value to avoid division by zero. Defaults to 1e-5.
    
    Attributes:
        eps (float): Small value for numerical stability
        embed_dim (int): Dimension of the input embeddings
        weight (nn.Parameter): Learnable scaling parameter
    """
    def __init__(self, embed_dim: int, eps: float = 1e-6, bias: bool=False):
        super().__init__()

        self.eps = eps
        self.scale = nn.Parameter(torch.ones(embed_dim))
        self.shift = nn.Parameter(torch.zeros(embed_dim)) if bias else None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for RMSNorm.
        
        Args:
            x (torch.Tensor): Input tensor of shape (..., dim)
            
        Returns:
            torch.Tensor: Normalized output tensor
        """
        # Match HF Gemma3: compute norm in float32, then scale by (1 + w)
        # input dtype
        input_dtype = x.dtype
        # compatible
        x = x.float()
        # Compute mean square with inplace operations for memory efficiency
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        # Use fused operation for better numerical stability
        x_norm = x * torch.rsqrt(variance + self.eps)
        out = x_norm * (1.0 + self.scale.float())

        if self.shift is not None:
            out = out + self.shift.float()

        return out.to(input_dtype)




# 测试代码 main 函数
def main():
    torch.manual_seed(123)

    # input
    example_batch = torch.randn(2, 3, 4)
    # RMSNorm
    rms_norm = RMSNorm(embed_dim=example_batch.shape[-1], eps=1e-5)
    rms_norm_pytorch = nn.RMSNorm(example_batch.shape[-1], eps=1e-5)
    rms_norm_qwen = RMSNorm_Qwen(embed_dim=example_batch.shape[-1], eps=1e-5)
    print(rms_norm(example_batch))
    print(rms_norm_pytorch(example_batch))
    print(rms_norm_qwen(example_batch))

    assert torch.allclose(rms_norm(example_batch), rms_norm_pytorch(example_batch)), "RMSNorm test failed"
    assert torch.allclose(rms_norm(example_batch), rms_norm_qwen(example_batch)), "RMSNorm_Qwen test failed"

if __name__ == "__main__":
    main()
