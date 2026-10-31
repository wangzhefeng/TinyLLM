# -*- coding: utf-8 -*-

# ***************************************************
# * File        : block_assembly.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2026-10-28
# * Version     : 1.0.102801
# * Description : Transformer Block 章节示例：手动组装 GPT-2（embed -> block x N
# *               -> final_norm -> out_head），与 models/gpt2 的 Model 参数量对账，
# *               并用同一组公式外推 124M 的参数构成。学习路线图阶段 1。
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)

from types import SimpleNamespace

import torch

from layers.normalization.layer_norm import LayerNorm
from layers.transformer_block import TransformerBlockGPT2_124M
from models.gpt2.gpt2_124M import Model
from utils.log_util import logger

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


def _cfg():
    return SimpleNamespace(
        vocab_size=128, context_length=16, embed_dim=32, d_ff=64,
        n_heads=4, n_layers=2, dropout=0.0, qkv_bias=True,
        dtype=torch.float32, is_train=False,
    )


def _n_params(module):
    return sum(p.numel() for p in module.parameters())


def manual_assembly():
    """手搭一个与 Model 相同结构的 GPT-2，逐组件统计参数量。"""
    torch.manual_seed(0)
    cfg = _cfg()

    tok_embed = torch.nn.Embedding(cfg.vocab_size, cfg.embed_dim)
    pos_embed = torch.nn.Embedding(cfg.context_length, cfg.embed_dim)
    blocks = torch.nn.ModuleList(
        [TransformerBlockGPT2_124M(cfg) for _ in range(cfg.n_layers)]
    )
    final_norm = LayerNorm(cfg.embed_dim)
    out_head = torch.nn.Linear(cfg.embed_dim, cfg.vocab_size, bias=False)

    parts = {
        "tok_embed": _n_params(tok_embed),
        "pos_embed": _n_params(pos_embed),
        f"trf_blocks x{cfg.n_layers}": _n_params(blocks),
        "final_norm": _n_params(final_norm),
        "out_head(未 tying)": _n_params(out_head),
    }
    for name, n in parts.items():
        logger.info(f"{name:>20}: {n:,}")
    manual_total = sum(parts.values())

    model = Model(cfg)
    model_total = _n_params(model)
    logger.info(f"{'手动组装合计':>20}: {manual_total:,}")
    logger.info(f"{'Model 实际参数':>20}: {model_total:,}")
    assert manual_total == model_total, "手动组装与 Model 参数量不一致"

    # 前向形状自检
    logits = model(torch.randint(0, cfg.vocab_size, (1, 8)))
    logger.info(f"Model 前向: (1, 8) ids -> {tuple(logits.shape)} logits")


def estimate_124m():
    """用同一组公式外推 GPT-2 124M 的参数构成（vocab=50257, ctx=1024, d=768,
    d_ff=3072, 12 层 12 头；out_head 与 tok_embed tying，不额外计数）。"""
    V, T, D, F, L = 50257, 1024, 768, 3072, 12
    wte = V * D
    wpe = T * D
    # 各投影带 bias（GPT-2 官方 qkv_bias=True）：W_q/W_k/W_v 与 out_proj
    attn = (D * D + D) * 3 + (D * D + D)
    ffn = (D * F + F) + (F * D + D)             # c_fc + c_proj，各带 bias
    ln = 2 * (D + D)                            # ln_1/ln_2（gamma+beta）
    block = attn + ffn + ln
    total = wte + wpe + L * block + (D + D)     # + ln_f；out_head 与 wte tying
    logger.info(f"wte: {wte:,}  wpe: {wpe:,}  block x{L}: {L * block:,}  ln_f: {2 * D:,}")
    logger.info(f"124M 估计: {total:,}（官方 124,439,808，tying 下 out_head 不计）")


# 测试代码 main 函数
def main():
    logger.info(f"{'-'*20} 手动组装 GPT-2（tiny） {'-'*20}")
    manual_assembly()
    logger.info(f"{'-'*20} 外推 124M 参数构成 {'-'*20}")
    estimate_124m()

if __name__ == "__main__":
    main()
