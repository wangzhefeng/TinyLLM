# -*- coding: utf-8 -*-

# ***************************************************
# * File        : final_norm.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-07-27
# * Version     : 1.1.072816
# * Description : FinalRMSNorm: 模型最后一层 decoder 之后的归一化层。
# *               与 RMSNorm 语义相同，单独命名以保持与
# *               Qwen3Model(embed -> layers -> final_norm) 的结构对应。
# * Link        : https://github.com/GeeeekExplorer/nano-vllm
# * Requirement : torch >= 2.0
# ***************************************************

__all__ = [
    "FinalRMSNorm",
]

# python libraries
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)
PKG_ROOT = str(Path(__file__).resolve().parents[1])
if PKG_ROOT in sys.path:
    sys.path.remove(PKG_ROOT)
sys.path.insert(0, PKG_ROOT)
import warnings
warnings.filterwarnings("ignore")

from layers.norm import RMSNorm

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


class FinalRMSNorm(RMSNorm):
    """
    最终的层归一化(语义与 RMSNorm 相同，仅用于区分网络位置)
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__(hidden_size, eps)




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
