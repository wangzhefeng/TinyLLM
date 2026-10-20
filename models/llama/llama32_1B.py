# -*- coding: utf-8 -*-

# ***************************************************
# * File        : llama32_1B.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-03-21
# * Version     : 2.0.102021
# * Description : Llama 3.2 1B。架构与 Llama 3 相同（GQA + RoPE + SwiGLU），
# *               差异：层数 16、rope_freq factor=32。
# *               注：官方 1B 权重还启用 tie_word_embeddings（out_head 与
# *               tok_embed 共享），教学实现未 tying，加载官方权重时需注意。
# * Link        : https://www.llama.com/docs/model-cards-and-prompt-formats/llama3_2
# * Requirement : torch
# ***************************************************

__all__ = ["Model"]

# python libraries
from pathlib import Path

from models.llama.llama3_8B import Model  # noqa: F401  复用 Llama3 实现

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


# 测试代码 main 函数
def main():
    import torch
    from utils.args_tools import DotDict
    from config.llama.model_cfgs import LLAMA32_CONFIG_1B
    from utils.model_memory import model_memory_size

    # model params
    cfg = DotDict(LLAMA32_CONFIG_1B)

    # model
    model = Model(cfg)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")   # ≈ 1.5B（未 tying；官方 tying 后 ≈ 1.24B）
    model_memory_size(model, input_dtype=torch.bfloat16)

if __name__ == "__main__":
    main()
