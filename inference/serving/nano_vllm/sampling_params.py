# -*- coding: utf-8 -*-

# ***************************************************
# * File        : sampling_params.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-07-28
# * Version     : 1.0.072816
# * Description : 采样参数(temperature / top_k / max_tokens / ignore_eos)，
# *               对应上游 nanovllm/sampling_params.py，额外支持 top_k。
# * Link        : https://github.com/GeeeekExplorer/nano-vllm
# * Requirement : torch >= 2.0
# ***************************************************

__all__ = [
    "SamplingParams",
]

# python libraries
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)
import warnings
warnings.filterwarnings("ignore")

from dataclasses import dataclass

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


@dataclass
class SamplingParams:
    """
    采样参数

    Args:
        temperature: 采样温度。0 表示贪心(argmax)，越大越随机
        top_k: 仅保留概率最高的 top_k 个 token 参与采样，-1 表示不截断
        max_tokens: 最多生成的 token 数
        ignore_eos: 是否忽略 eos(即使采样到 eos 也继续生成，直到 max_tokens)
    """
    temperature: float = 1.0
    top_k: int = -1
    max_tokens: int = 64
    ignore_eos: bool = False

    def __post_init__(self):
        assert self.temperature >= 0.0, "temperature 必须 >= 0 (0 表示贪心)"
        assert self.top_k == -1 or self.top_k >= 1
        assert self.max_tokens >= 1




# 测试代码 main 函数
def main():
    sp = SamplingParams(temperature=0.8, top_k=5, max_tokens=16)
    print(sp)

if __name__ == "__main__":
    main()
