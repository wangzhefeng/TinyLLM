# -*- coding: utf-8 -*-

# ***************************************************
# * File        : model.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2026-10-20
# * Version     : 2.0.102020
# * Description : DeepSeek 风格 decoder-only 模型组装（教学版）：
# *               MLA 注意力 + RMSNorm + SwiGLU FFN，支持 KV cache 增量解码。
# *               完整 DeepSeek-V3 还需要 DeepSeekMoE（shared + routed experts，
# *               见 layers/moe.py 的扩展点）与 MTP（多 token 预测，未实现）。
# * Link        : https://github.com/deepseek-ai/DeepSeek-V3
# ***************************************************

__all__ = [
    "DeepSeekBlock",
    "DeepSeekModel",
]

# python libraries
from pathlib import Path

import torch
import torch.nn as nn

from layers.feed_forward import FeedForwardSiLU
from layers.normalization.rms_norm import RMSNorm
from layers.kv_cache import KVCache
from models.deepseek.mla import MultiHeadLatentAttention

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


class DeepSeekBlock(nn.Module):
    """pre-LN block：RMSNorm -> MLA -> residual -> RMSNorm -> SwiGLU FFN -> residual。"""

    def __init__(self, cfg):
        super().__init__()

        self.attn = MultiHeadLatentAttention(
            d_model=cfg.embed_dim,
            n_heads=cfg.n_heads,
            kv_lora_rank=cfg.kv_lora_rank,
            q_lora_rank=cfg.q_lora_rank,
            qk_nope_head_dim=cfg.qk_nope_head_dim,
            qk_rope_head_dim=cfg.qk_rope_head_dim,
            rope_base=cfg.rope_base,
            context_length=cfg.context_length,
            dtype=cfg.dtype,
        )
        self.ff = FeedForwardSiLU(cfg)
        self.norm1 = RMSNorm(cfg.embed_dim, eps=1e-6)
        self.norm2 = RMSNorm(cfg.embed_dim, eps=1e-6)

    def forward(self, x, start_pos=0, cache=None):
        shortcut = x
        x = self.norm1(x)
        x, next_cache = self.attn(x, start_pos=start_pos, cache=cache)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = x + shortcut

        return x, next_cache


class DeepSeekModel(nn.Module):
    """DeepSeek-V2/V3 风格模型（教学小配置版）。

    forward 协议与 Qwen3 一致：``forward(x, cache=None)``，
    ``cache`` 为 ``layers.kv_cache.KVCache``（每层存 MLA 的 ``(c_kv, k_rope)`` 元组），
    ``reset_kv_cache()`` 复位位置计数。
    """

    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.tok_embed = nn.Embedding(cfg.vocab_size, cfg.embed_dim, dtype=cfg.dtype)
        self.trf_blocks = nn.ModuleList([DeepSeekBlock(cfg) for _ in range(cfg.n_layers)])
        self.final_norm = RMSNorm(cfg.embed_dim, eps=1e-6)
        self.out_head = nn.Linear(cfg.embed_dim, cfg.vocab_size, bias=False, dtype=cfg.dtype)
        self.current_pos = 0

    def forward(self, x, cache=None):
        num_tokens = x.shape[1]
        if cache is not None:
            pos_start = self.current_pos
            pos_end = pos_start + num_tokens
            if pos_end > self.cfg.context_length:
                raise IndexError(
                    f"KV cache position {pos_end} exceeds context_length "
                    f"{self.cfg.context_length}: 请调用 reset_kv_cache() 或减小生成长度。"
                )
            self.current_pos = pos_end
        else:
            pos_start = 0

        x = self.tok_embed(x)
        for i, block in enumerate(self.trf_blocks):
            block_cache = cache.get(i) if cache is not None else None
            x, new_block_cache = block(x, start_pos=pos_start, cache=block_cache)
            if cache is not None:
                cache.update(i, new_block_cache)

        x = self.final_norm(x)
        return self.out_head(x.to(self.cfg.dtype))

    def reset_kv_cache(self):
        self.current_pos = 0


# MTP（Multi-Token Prediction）为 V3 训练专用组件，教学版暂不实现；
# 需要时参考 https://arxiv.org/abs/2412.19437 §2.5.1 与 docs/40-models/deepseek.md。

# 测试代码 main 函数
def main():
    import torch

    from utils.args_tools import DotDict
    from utils.log_util import logger

    # DeepSeek-tiny 教学配置（V3 结构缩小 ~200 倍）
    DEEPSEEK_TINY_CONFIG = {
        "vocab_size": 1000,
        "context_length": 256,
        "embed_dim": 256,
        "n_layers": 4,
        "n_heads": 8,
        "kv_lora_rank": 64,           # V3: 512
        "q_lora_rank": 128,           # V3: 1536
        "qk_nope_head_dim": 24,       # V3: 128
        "qk_rope_head_dim": 8,        # V3: 64
        "hidden_dim": 512,            # FFN 中间维（FeedForwardSiLU 回退键）
        "rope_base": 10_000.0,
        "dtype": torch.bfloat16,
    }
    cfg = DotDict(DEEPSEEK_TINY_CONFIG)

    torch.manual_seed(123)
    model = DeepSeekModel(cfg).eval()
    x = torch.randint(0, cfg["vocab_size"], (2, 8))
    with torch.no_grad():
        logits = model(x)
    logger.info(f"forward: {tuple(logits.shape)}")

    # 参数量与 MLA cache 节省对比见 tests/unit/test_mla.py
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"params: {total:,}")

if __name__ == "__main__":
    main()
