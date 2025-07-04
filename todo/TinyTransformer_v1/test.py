# -*- coding: utf-8 -*-

# ***************************************************
# * File        : test.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2024-09-24
# * Version     : 1.0.092404
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# * TODO        : 1.
# ***************************************************

__all__ = []

# python libraries
import os
import sys
ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
from dataclasses import dataclass

import torch
from tiny_transformer import Transformer

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


# ------------------------------
# 创建模型配置文件
# ------------------------------ 
print("*" * 80)
@dataclass
class TransformerConfig:
    block_size: int = 1024  # 序列的最大长度
    vocab_size: int = 50304  # 词表大小
    n_layer: int = 4  # Encoder, Deocder 层数
    n_head: int = 4  # 注意力头数量
    n_embd: int = 768  # Embedding 维度(d_model)
    dropout: float = 0.0
    bias: bool = True

model_config = TransformerConfig(
    block_size = 12,
    vocab_size = 10,
    n_layer = 2,
    n_head = 4,
    n_embd = 16,
    dropout = 0.0,
    bias = True,
)
print(f"Model Config:\n{model_config}")

# ------------------------------
# 创建模型
# ------------------------------
print("*" * 80)
print("Model info:")
model = Transformer(model_config)

# ------------------------------
# 向前传递
# ------------------------------
print("*" * 80)
idx = torch.randint(1, 10, (4, 8))
print(f"Model input:\n{idx}")
print(f"Model input size:\n{idx.size()}")
print("-" * 45)
print("Model forward:")
logits, loss = model(idx, targets = None)
print("-" * 45)
print(f"logits size:\n{logits.size()}")
print(f"logits:\n{logits}")

# ------------------------------
# 模型推理
# ------------------------------ 
print("*" * 80)
print("Model Inference:")
result = model.generate(idx, 3)
print("-" * 45)
print(f"generate result:\n{result}")
print(f"generate result size:\n{result.size()}")




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
