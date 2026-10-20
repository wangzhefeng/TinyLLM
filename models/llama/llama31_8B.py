# -*- coding: utf-8 -*-

# ***************************************************
# * File        : llama31_8B.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-04-04
# * Version     : 2.0.102021
# * Description : Llama 3.1 8B。架构与 Llama 3 相同（GQA + RoPE + SwiGLU），
# *               唯一差异是 rope_freq 频率缩放（factor=8，长上下文 128k），
# *               因此直接复用 models/llama/llama3_8B.py 的 Model。
# * Link        : https://www.llama.com/docs/model-cards-and-prompt-formats/llama3_1
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
    from config.llama.model_cfgs import LLAMA31_CONFIG_8B
    from utils.model_memory import model_memory_size

    # model params
    cfg = DotDict(LLAMA31_CONFIG_8B)

    # model
    model = Model(cfg)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")   # ≈ 8B
    model_memory_size(model, input_dtype=torch.bfloat16)

if __name__ == "__main__":
    main()
