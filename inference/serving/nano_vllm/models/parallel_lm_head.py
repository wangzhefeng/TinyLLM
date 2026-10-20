# -*- coding: utf-8 -*-

# ***************************************************
# * File        : parallel_lm_head.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-07-27
# * Version     : 1.1.072816
# * Description : ParallelLMHead 语言模型头。
# *               按词表维度做张量并行，world_size=1 时等价于
# *               nn.Linear(hidden_size, vocab_size, bias=False)。
# * Link        : https://github.com/GeeeekExplorer/nano-vllm
# * Requirement : torch >= 2.0
# ***************************************************

__all__ = [
    "ParallelLMHead",
]

# python libraries
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from ..layers.embed import VocabParallelEmbedding

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


class ParallelLMHead(VocabParallelEmbedding):
    """
    语言模型头，用于生成最终的输出 logits。

    权重布局与 VocabParallelEmbedding 相同(按词表切分)，
    forward 用 F.linear 计算 logits；world_size>1 时各 rank
    计算本地词表的 logits 后 all-gather 拼接(本实现仅支持 world_size=1)。
    """

    def __init__(self, num_embed: int, embed_dim: int):
        super().__init__(num_embed, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 本地词表的 logits
        logits = F.linear(x, self.weight)
        if self.tp_size > 1:
            # 收集所有 rank 的 logits 并沿词表维度拼接。
            # 注意：all_gather 的输出写入发生在 gloo 工作线程(inference mode 之外)，
            # 因此接收缓冲区必须显式创建为普通 tensor，不能是 inference tensor
            with torch.inference_mode(False):
                all_logits = [torch.empty_like(logits) for _ in range(self.tp_size)]
            dist.all_gather(all_logits, logits)
            all_logits[self.tp_rank] = logits
            logits = torch.cat(all_logits, dim=-1)
        return logits




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
