# -*- coding: utf-8 -*-

# ***************************************************
# * File        : pos_encoding.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2026-10-28
# * Version     : 1.0.102801
# * Description : 位置编码章节示例：GPT-2 的可学习位置嵌入 vs 正弦固定位置编码。
# *               学习路线图阶段 1。
# * Link        : paper: https://arxiv.org/abs/1706.03762
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)

import torch

from layers.position_encoding.FixPE import PositionalEncoding
from utils.log_util import logger

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


def learned_position_embedding():
    """GPT-2 方案：位置嵌入是一个可学习的 nn.Embedding（有参数）。"""
    vocab_size, context_length, embed_dim = 50257, 1024, 768
    pos_embed = torch.nn.Embedding(context_length, embed_dim)
    n_params = pos_embed.weight.numel()
    logger.info(f"learned PE: {context_length} x {embed_dim} = {n_params:,} 个可训练参数")
    ids = torch.arange(6)
    out = pos_embed(ids)
    logger.info(f"前 6 个位置 -> 形状 {tuple(out.shape)}")


def sinusoidal_position_encoding():
    """原始 Transformer 方案：正弦/余弦固定编码（无参数，任意长度可外推）。"""
    d_model, max_len = 768, 5000
    pe = PositionalEncoding(d_model=d_model, max_len=max_len)
    n_params = sum(p.numel() for p in pe.parameters())
    logger.info(f"sinusoidal PE: 可训练参数 {n_params}（固定表，max_len={max_len}）")
    x = torch.zeros(1, 16, d_model)
    out = pe(x)
    logger.info(f"前 16 个位置 -> 形状 {tuple(out.shape)}")
    # 位置编码的余弦相似度随距离衰减——模型由此感知相对位置
    rows = out[0]
    for dist in (1, 2, 4, 8):
        sim = torch.nn.functional.cosine_similarity(rows[0], rows[dist], dim=0).item()
        logger.info(f"cos(pe[0], pe[{dist}]) = {sim:.3f}")


# 测试代码 main 函数
def main():
    logger.info(f"{'-'*20} learned PE（GPT-2） {'-'*20}")
    learned_position_embedding()
    logger.info(f"{'-'*20} sinusoidal PE（Transformer 2017） {'-'*20}")
    sinusoidal_position_encoding()

if __name__ == "__main__":
    main()
