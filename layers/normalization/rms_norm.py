import torch
import torch.nn as nn

__all__ = ["RMSNorm", "RMSNorm_Qwen3", "RMSNorm_Gemma3"]


class _RMSNorm(nn.Module):
    def __init__(self, embed_dim, eps, bias=False, compatible=True, zero_init_gamma=False):
        super().__init__()
        self.eps = eps
        self.embed_dim = embed_dim
        self.compatible = compatible
        self.zero_init_gamma = zero_init_gamma
        self.scale = nn.Parameter(torch.ones(embed_dim))
        self.shift = nn.Parameter(torch.zeros(embed_dim)) if bias else None

    def forward(self, x):
        input_dtype = x.dtype
        normalized = x.float() if self.compatible else x
        variance = normalized.pow(2).mean(dim=-1, keepdim=True)
        normalized = normalized * torch.rsqrt(variance + self.eps)
        scale = 1.0 + self.scale if self.zero_init_gamma else self.scale
        normalized = normalized * scale
        if self.shift is not None:
            normalized = normalized + self.shift
        return normalized.to(input_dtype)


class RMSNorm(nn.Module):
    def __init__(self, embed_dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.embed_dim = embed_dim
        self.weight = nn.Parameter(torch.ones(embed_dim)).float()

    def forward(self, x):
        input_dtype = x.dtype
        normalized = x.float()
        variance = normalized.pow(2).mean(dim=-1, keepdim=True)
        normalized = normalized * torch.rsqrt(variance + self.eps)
        return (normalized * self.weight.float()).to(input_dtype)


class RMSNorm_Qwen3(_RMSNorm):
    def __init__(self, embed_dim: int, eps: float = 1e-6, bias: bool = False, compatible: bool = True):
        super().__init__(embed_dim, eps, bias=bias, compatible=compatible)


class RMSNorm_Gemma3(_RMSNorm):
    def __init__(self, embed_dim: int, eps: float = 1e-6, bias: bool = False):
        super().__init__(embed_dim, eps, bias=bias, zero_init_gamma=True)
