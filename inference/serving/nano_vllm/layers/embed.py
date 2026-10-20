# -*- coding: utf-8 -*-

# ***************************************************
# * File        : embed.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-07-27
# * Version     : 1.1.072816
# * Description : 词嵌入层
# *               VocabParallelEmbedding: 按词表维度做张量并行的 Embedding。
# *               torch.distributed 未初始化时退化为 world_size=1 的普通 Embedding。
# * Link        : https://github.com/GeeeekExplorer/nano-vllm
# * Requirement : torch >= 2.0
# ***************************************************

__all__ = [
    "VocabParallelEmbedding",
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

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from layers.parallel_linear import _tp_rank_and_size

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


class VocabParallelEmbedding(nn.Module):
    """
    词嵌入层

    在自然语言处理（NLP）和机器学习中，Embedding 是一种将离散的类别变量
    （例如词汇表中的单词）转换为连续的、低维度的向量表示的技术。

    Embedding 就是将每个单词映射到一个由实数组成的向量。
    它的核心作用是捕捉词与词之间的语义关系。

    在一个训练良好的模型中，意思相近的词，
    其对应的 Embedding 向量在向量空间中的距离也更近。

    TP 语义：词表按 rank 切分为 tp_size 份，每个 rank 只保存自己的分片；
    forward 时用 mask 选出属于本分片的 token，本地查找后 all-reduce 求和。
    world_size=1 时整个词表都在本地，等价于 nn.Embedding。
    """
    def __init__(self, num_embed: int, embed_dim: int):
        super().__init__()

        # 获取当前进程的排名(rank)和总进程数(world_size)，未初始化 dist 时为 (0, 1)
        self.tp_rank, self.tp_size = _tp_rank_and_size()
        assert num_embed % self.tp_size == 0
        self.num_embed = num_embed
        # 计算每个分区的大小，以及当前进程负责的词汇表的起始和结束索引
        self.num_embed_per_partition = num_embed // self.tp_size
        self.vocab_start_idx = self.num_embed_per_partition * self.tp_rank
        self.vocab_end_idx = self.vocab_start_idx + self.num_embed_per_partition
        # 只为当前分区创建权重参数，而不是整个词汇表
        self.weight = nn.Parameter(torch.empty(self.num_embed_per_partition, embed_dim))
        self.weight.weight_loader = self.weight_loader

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(0)
        # 计算在完整权重张量中的起始位置
        start_idx = self.tp_rank * shard_size
        # 使用 narrow 方法精确地切出所需的分片
        loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)
        assert param_data.size() == loaded_weight.size()
        # 将切片后的权重拷贝到参数中
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor):
        if self.tp_size > 1:
            # 生成掩码 (Mask)，判断输入 Token 是否属于当前分区
            mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx)
            # 转换本地索引，并用掩码清零无关 Token
            x = mask * (x - self.vocab_start_idx)
        # 本地 Embedding 查找
        y = F.embedding(x, self.weight)

        if self.tp_size > 1:
            # 清零无关输出，确保只保留当前分区贡献的向量
            y = mask.unsqueeze(1) * y
            # 聚合 (All-Reduce)，将所有 rank 的结果相加
            dist.all_reduce(y)

        return y




# 测试代码 main 函数
def main():
    # world_size=1(dist 未初始化) 时等价于 nn.Embedding
    torch.manual_seed(0)
    emb = VocabParallelEmbedding(12, 4)
    ref = nn.Embedding(12, 4)
    with torch.no_grad():
        ref.weight.copy_(emb.weight)
    ids = torch.tensor([1, 5, 11])
    assert torch.allclose(emb(ids), ref(ids))
    print("VocabParallelEmbedding world_size=1 退化测试通过")

if __name__ == "__main__":
    main()
