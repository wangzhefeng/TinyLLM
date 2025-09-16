# -*- coding: utf-8 -*-

# ***************************************************
# * File        : Gemma3_270M.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-09-13
# * Version     : 1.0.091322
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
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn

from layers.transformer_block import TransformerBlockGemma3
from layers.normailzation.rms_norm import RMSNorm_Gemma3
from layers.position_encoding.RoPE import precompute_rope_params, compute_rope

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


class Model(nn.Module):
    
    def __init__(self, cfg):
        super().__init__()

        assert cfg.layer_types is not None and len(cfg.layer_types) == cfg.n_layers

        self.cfg = cfg
        # main model parameters
        self.tok_embed = nn.Embedding(cfg.vocab_size, cfg.embed_dim, dtype=cfg.dtype)
        self.blocks = nn.ModuleList([
            TransformerBlockGemma3(cfg, attn_type) for attn_type in cfg.layer_types
        ])
        self.final_norm = RMSNorm_Gemma3(cfg.embed_dim, eps=1e-6)
        self.out_head = nn.Linear(cfg.embed_dim, cfg.vocab_size, bias=False, dtype=cfg.dtype)
        # reusable utilities
        cos_local, sin_local = precompute_rope_params(
            head_dim=cfg.head_dim,
            theta_base=cfg.rope_local_base,
            context_length=cfg.context_length,
            dtype=torch.float32,
        )
        cos_global, sin_global = precompute_rope_params(
            head_dim=cfg.head_dim,
            theta_base=cfg.rope_base,
            context_length=cfg.context_length,
            dtype=torch.float32,
        )
        self.register_buffer("cos_local", cos_local, persistent=False)
        self.register_buffer("sin_local", sin_local, persistent=False)
        self.register_buffer("cos_global", cos_global, persistent=False)
        self.register_buffer("sin_global", sin_global, persistent=False)

    def _create_masks(self, seq_len, device):
        ones = torch.ones((seq_len, seq_len), dtype=torch.bool, device=device)
    
        # mask_global (future is masked: j > i)
        #     j:  0 1 2 3 4 5 6 7
        #  i
        #     0:  0 1 1 1 1 1 1 1
        #     1:  0 0 1 1 1 1 1 1
        #     2:  0 0 0 1 1 1 1 1
        #     3:  0 0 0 0 1 1 1 1
        #     4:  0 0 0 0 0 1 1 1
        #     5:  0 0 0 0 0 0 1 1
        #     6:  0 0 0 0 0 0 0 1
        #     7:  0 0 0 0 0 0 0 0
        mask_global = torch.triu(ones, diagonal=1)
    
        # far_past (too far back is masked: i - j >= sliding_window)
        # where sliding_window = 4
        #     j:  0 1 2 3 4 5 6 7
        #  i
        #     0:  0 0 0 0 0 0 0 0
        #     1:  0 0 0 0 0 0 0 0
        #     2:  0 0 0 0 0 0 0 0
        #     3:  0 0 0 0 0 0 0 0
        #     4:  1 0 0 0 0 0 0 0
        #     5:  1 1 0 0 0 0 0 0
        #     6:  1 1 1 0 0 0 0 0
        #     7:  1 1 1 1 0 0 0 0
        far_past = torch.triu(ones, diagonal=self.cfg.sliding_window).T
    
        # Local (sliding_window) = future OR far-past
        # mask_local
        #     j:  0 1 2 3 4 5 6 7
        # i
        # 0:      0 1 1 1 1 1 1 1
        # 1:      0 0 1 1 1 1 1 1
        # 2:      0 0 0 1 1 1 1 1
        # 3:      0 0 0 0 1 1 1 1
        # 4:      1 0 0 0 0 1 1 1
        # 5:      1 1 0 0 0 0 1 1
        # 6:      1 1 1 0 0 0 0 1
        # 7:      1 1 1 1 0 0 0 0
        mask_local = mask_global | far_past
        
        return mask_global, mask_local

    def forward(self, x):
        # Forward pass
        batch_size, seq_len = x.shape
        x = self.tok_embed(x) * (self.cfg.embed_dim ** 0.5)
        mask_global, mask_local = self._create_masks(seq_len, x.device)
        for block in self.blocks:
            x = block(
                x, 
                mask_global=mask_global,
                mask_local=mask_local,
                cos_global=self.cos_global,
                sin_global=self.sin_global,
                cos_local=self.cos_local,
                sin_local=self.sin_local,
            )
        x = self.final_norm(x)
        logits = self.out_head(x.to(self.cfg.dtype))

        return logits







# 测试代码 main 函数
def main():
    from utils.log_util import logger

    # ------------------------------
    # parms
    # ------------------------------
    GEMMA3_CONFIG_270M = {
        "vocab_size": 262_144,
        "context_length": 32_768,
        "embed_dim": 640,
        "n_heads": 4,
        "n_layers": 18,
        "hidden_dim": 2048,
        "d_ff": 2048,
        "head_dim": 256,
        "qk_norm": True,
        "n_kv_groups": 1,
        "rope_local_base": 10_000.0,
        "rope_base": 1_000_000.0,
        "sliding_window": 512,
        "layer_types": [
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "full_attention",
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "full_attention",
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "full_attention",
        ],
        "dtype": torch.bfloat16,
        "query_pre_attn_scalar": 256,
    }
    from utils.args_tools import DotDict
    GEMMA3_CONFIG_270M = DotDict(GEMMA3_CONFIG_270M)
    # ------------------------------
    # random seed
    # ------------------------------
    torch.manual_seed(123)
    # ------------------------------
    # model
    # ------------------------------
    model = Model(GEMMA3_CONFIG_270M)
    logger.info(model)
    # ------------------------------
    # model test
    # ------------------------------
    model_output = model(torch.tensor([1, 2, 3]).unsqueeze(0))
    logger.info(f"model_output: \n{model_output} \nmodel_output.shape: {model_output.shape}")
    # ------------------------------
    # model memory
    # ------------------------------
    # Account of model parameters
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total number of parameters: {total_params:,}")
    # Account for weight tying
    total_params_normalized = total_params - model.tok_embed.weight.numel()
    logger.info(f"Total number of unique parameters: {total_params_normalized:,}")
    from utils.model_memory import model_memory_size
    model_memory_size(model, input_dtype=torch.float32)
    model_memory_size(model, input_dtype=torch.bfloat16)

if __name__ == "__main__":
    main()
