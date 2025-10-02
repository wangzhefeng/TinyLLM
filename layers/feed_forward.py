# -*- coding: utf-8 -*-

# ***************************************************
# * File        : feed_forward.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-02-12
# * Version     : 1.0.021221
# * Description : description
# * Link        : link
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

from layers.activation import (
    ReLU, 
    ReLUPyTorch,
    GELU, 
    SiLU,
    SwiGLU,
)

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


class FeedForwardReLU(nn.Module):
    
    def __init__(self, cfg):
        super().__init__()

        self.fc1 = nn.Linear(cfg.embed_dim, cfg.d_ff, dtype=cfg.dtype, bias=True)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(cfg.d_ff, cfg.embed_dim, dtype=cfg.dtype, bias=True)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        out = self.fc2(x)

        return out


class FeedForwardGELU(nn.Module):
    
    def __init__(self, cfg):
        super().__init__()

        self.fc1 = nn.Linear(cfg.embed_dim, cfg.d_ff, dtype=cfg.dtype, bias=True)
        self.gelu = nn.GELU(approximate="tanh")
        self.fc2 = nn.Linear(cfg.d_ff, cfg.embed_dim, dtype=cfg.dtype, bias=True)
    
    def forward(self, x):
        # input tensor x.shape: [batch_size, num_tokens, embed_dim]
        # Linear layer
        x = self.fc1(x)  # [batch_size, num_tokens, 4*embed_dim]
        # GELU activation
        x = self.gelu(x)  # [batch_size, num_tokens, 4*embed_dim]
        # Linear layer
        out = self.fc2(x)  # [batch_size, num_tokens, embed_dim]

        return out


class FeedForwardGELU_Gemma3(nn.Module):
    
    def __init__(self, cfg):
        super().__init__()

        self.fc1 = nn.Linear(cfg.embed_dim, cfg.d_ff, dtype=cfg.dtype, bias=False)
        self.fc2 = nn.Linear(cfg.embed_dim, cfg.d_ff, dtype=cfg.dtype, bias=False)
        self.fc3 = nn.Linear(cfg.d_ff, cfg.embed_dim, dtype=cfg.dtype, bias=False)
        self.gelu = nn.GELU(approximate="tanh")
    
    def forward(self, x):
        # input tensor x.shape: [batch_size, num_tokens, embed_dim]
        # Linear layer
        x_fc1 = self.fc1(x)  # [batch_size, num_tokens, 4*embed_dim]
        x_fc2 = self.fc2(x)  # [batch_size, num_tokens, 4*embed_dim]
        # GELU activation
        x = self.gelu(x_fc1) * x_fc2  # [batch_size, num_tokens, 4*embed_dim]
        # Linear layer
        out = self.fc3(x)  # [batch_size, num_tokens, embed_dim]

        return out


class FeedForwardSiLU(nn.Module):
    """
    SwiGLU: GLU Variants Improve Transformer (2020): https://arxiv.org/abs/2002.05202
    """
    def __init__(self, cfg):
        super().__init__()

        self.fc1 = nn.Linear(cfg.embed_dim, cfg.d_ff, dtype=cfg.dtype, bias=False)
        self.fc2 = nn.Linear(cfg.embed_dim, cfg.d_ff, dtype=cfg.dtype, bias=False)
        self.silu = nn.SiLU()
        self.fc3 = nn.Linear(cfg.d_ff, cfg.embed_dim, dtype=cfg.dtype, bias=False)

    def forward(self, x):
        x_fc1 = self.fc1(x)
        x_fc2 = self.fc2(x)
        x = self.silu(x_fc1) * x_fc2
        out = self.fc3(x)

        return out


class MoEFeedForward(nn.Module):
    
    def __init__(self, cfg):
        super().__init__()
        
        self.num_experts_per_tok = cfg.num_experts_per_tok
        self.num_experts = cfg.num_experts
        self.embed_dim = cfg.embed_dim
        self.gate = nn.Linear(cfg.embed_dim, cfg.num_experts, bias=False, dtype=cfg.dtype)

        self.fc1 = nn.ModuleList([
            nn.Linear(cfg.embed_dim, cfg.moe_intermediate_size, bias=False, dtype=cfg.dtype)
            for _ in range(cfg.num_experts)
        ])
        self.fc2 = nn.ModuleList([
            nn.Linear(cfg.embed_dim, cfg.moe_intermediate_size, bias=False, dtype=cfg.dtype)
            for _ in range(cfg.num_experts)
        ])
        self.fc3 = nn.ModuleList([
            nn.Linear(cfg.moe_intermediate_size, cfg.embed_dim, bias=False, dtype=cfg.dtype)
            for _ in range(cfg.num_experts)
        ])

    def forward(self, x):
        scores = self.gate(x)  # (b, seq_len, num_experts)
        topk_scores, topk_indices = torch.topk(scores, self.num_experts_per_tok, dim=-1)
        topk_probs = torch.softmax(topk_scores, dim=-1)

        batch, seq_len, _ = x.shape
        x_flat = x.reshape(batch * seq_len, -1)
        out_flat = torch.zeros(batch * seq_len, self.embed_dim, device=x.device, dtype=x.dtype)

        topk_indices_flat = topk_indices.reshape(-1, self.num_experts_per_tok)
        topk_probs_flat = topk_probs.reshape(-1, self.num_experts_per_tok)

        unique_experts = torch.unique(topk_indices_flat)

        for expert_id_tensor in unique_experts:
            expert_id = int(expert_id_tensor.item())
            mask = topk_indices_flat == expert_id
            if not mask.any():
                continue

            token_mask = mask.any(dim=-1)
            selected_idx = token_mask.nonzero(as_tuple=False).squeeze(-1)
            if selected_idx.numel() == 0:
                continue

            expert_input = x_flat.index_select(0, selected_idx)
            hidden = torch.nn.functional.silu(self.fc1[expert_id](expert_input)) * self.fc2[expert_id](expert_input)
            expert_out = self.fc3[expert_id](hidden)

            mask_selected = mask[selected_idx]
            slot_indices = mask_selected.int().argmax(dim=-1, keepdim=True)
            selected_probs = torch.gather(topk_probs_flat.index_select(0, selected_idx), dim=-1, index=slot_indices).squeeze(-1)

            out_flat.index_add_(0, selected_idx, expert_out * selected_probs.unsqueeze(-1))

        return out_flat.reshape(batch, seq_len, self.embed_dim)


class FeedForwardSwiGLU(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.swiglu = None

    def forward(self, x):
        pass




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
