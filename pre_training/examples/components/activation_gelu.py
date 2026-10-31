# -*- coding: utf-8 -*-

# ***************************************************
# * File        : activation_gelu.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2026-10-28
# * Version     : 1.0.102801
# * Description : 激活函数章节示例：GELU 精确版与 tanh 近似版的数值差异，
# *               以及它为什么影响与 HF 官方实现的逐位对齐（GPT-2 用 tanh 版）。
# *               学习路线图阶段 1。
# * Link        : paper: https://arxiv.org/abs/1606.08415
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)

import torch

from layers.feed_forward import FeedForwardGELU
from utils.log_util import logger

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


def gelu_exact_vs_tanh():
    x = torch.linspace(-4, 4, 9)
    exact = torch.nn.functional.gelu(x)                 # 精确版（默认）
    tanh = torch.nn.functional.gelu(x, approximate="tanh")
    logger.info(f"x:      {[f'{v: .2f}' for v in x]}")
    logger.info(f"exact:  {[f'{v: .4f}' for v in exact]}")
    logger.info(f"tanh:   {[f'{v: .4f}' for v in tanh]}")
    diff = (exact - tanh).abs()
    logger.info(f"|diff|: {[f'{v: .4f}' for v in diff]}  max={diff.max():.4f}")
    logger.info("GPT-2 用 tanh 版；加载官方权重做逐位对齐时必须一致（见 test_gpt2_hf_alignment）")


def ffn_uses_tanh():
    """仓库的 GPT-2 FFN 显式使用 approximate='tanh'。"""
    from types import SimpleNamespace

    torch.manual_seed(0)
    cfg = SimpleNamespace(embed_dim=4, d_ff=8, dtype=torch.float32)
    ffn = FeedForwardGELU(cfg)
    out = ffn(torch.randn(2, 4))
    logger.info(f"FeedForwardGELU 前向形状: {tuple(out.shape)}（GELU(tanh) 在中间层）")


# 测试代码 main 函数
def main():
    logger.info(f"{'-'*20} GELU 精确版 vs tanh 近似版 {'-'*20}")
    gelu_exact_vs_tanh()
    ffn_uses_tanh()

if __name__ == "__main__":
    main()
