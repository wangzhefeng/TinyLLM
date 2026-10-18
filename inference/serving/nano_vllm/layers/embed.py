# -*- coding: utf-8 -*-

# ***************************************************
# * File        : embed.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-07-27
# * Version     : 1.0.072716
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

__all__ = []

# python libraries
import os
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]
os.environ['LOG_NAME'] = LOGGING_LABEL
from utils.log_util import logger


class VocabParallelEmbedding(nn.Embedding):
    """
    词嵌入层
    
    在自然语言处理（NLP）和机器学习中，Embedding 是一种将离散的类别变量
    （例如词汇表中的单词）转换为连续的、低维度的向量表示的技术。

    Embedding 就是将每个单词映射到一个由实数组成的向量。
    它的核心作用是捕捉词与词之间的语义关系。 

    在一个训练良好的模型中，意思相近的词，
    其对应的 Embedding 向量在向量空间中的距离也更近。
    """
    def __init__(self, num_embed: int, embed_dim: int):
        super().__init__()

        assert num_embed % self.tp_size == 0
        # 获取当前进程的排名(rank)和总进程数(world_size)
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
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
            # 生成掩码 (Mask)，判断输入Token是否属于当前分区
            mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx)
            # 转换本地索引，并用掩码清零无关Token
            x = mask * (x - self.vocab_start_idx)
        # 本地 Embedding 查找
        y = F.embedding(x, self.weight)

        if self.tp_size > 1:
            # 清零无关输出，确保只保留当前分区贡献的向量
            y = mask.unsqueeze(1) * y
            # 聚合 (All-Reduce)，将所有GPU的结果相加
            dist.all_reduce(y)
        
        return y


class VocabParallelEmbedding_test(nn.Module):
    
    def __init__(self, num_embed: int, embed_dim: int, tp_rank: int, tp_size: int):
        super().__init__()

        assert num_embed % tp_size == 0

        self.num_embed = num_embed
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.num_embed_per_partition = num_embed // tp_size
        self.vocab_start_idx = self.num_embed_per_partition * tp_rank
        self.vocab_end_idx = self.vocab_start_idx + self.num_embed_per_partition 
        self.weight = nn.Parameter(torch.empty(self.num_embed_per_partition, embed_dim))
        logger.info(f"self.num_embed_per_partition: {self.num_embed_per_partition}")
        logger.info(f"self.vocab_start_idx: {self.vocab_start_idx}")
        logger.info(f"self.vocab_end_idx: {self.vocab_end_idx}")
        logger.info(f"self.weight: \n{self.weight}")
    
    def forward(self, x: torch.Tensor):
        # masking
        mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx)
        logger.info(f"[步骤 1: 生成的掩码(Mask)]\n{mask}")
        # local index
        local_x = (x - self.vocab_start_idx) * mask
        logger.info(f"[步骤 2: 转换后的本地索引]\n{local_x}")
        # embedding
        y = F.embedding(local_x, self.weight)
        logger.info(f"[步骤 3: 本地 Embedding 查找结果]\n{y}")
        # output
        y = y * mask.unsqueeze(-1)
        logger.info(f"[步骤 4: 清零后的输出(准备 All-Reduce)]\n{y}")




# 测试代码 main 函数
def main():
    # ------------------------------
    # Vocabulary
    # ------------------------------
    # 注意：为了演示方便，"a" 在词汇表里出现了两次，分别在两个分片中。在真实场景中，词汇表中的词是唯一的。
    VOCAB = [
        "a", "big", "cat", "sits", "on", "the",   # 这 6 个词属于 Rank 0
        "mat", "and", "a", "small", "dog", "plays"# 这 6 个词属于 Rank 1
    ]
    # ------------------------------
    # Tokenizer 
    # ------------------------------
    word_to_id = {word: i for i, word in enumerate(VOCAB)}
    id_to_word = {i: word for i, word in enumerate(VOCAB)}
    # ------------------------------
    # Input
    # ------------------------------
    # 输入从 token 编码为 token_id
    text_input = ["cat", "dog"]
    input_ids = torch.tensor([[word_to_id[word] for word in text_input]], dtype=torch.long)
    logger.info(f"input_ids: {input_ids} {input_ids.dtype}")
    # ------------------------------
    # 
    # ------------------------------
    # params
    VOCAB_SIZE = len(VOCAB)
    EMBEDDING_DIM = 4
    TP_SIZE = 2

    logger.info("="*60) 
    logger.info(f"模拟开始: TP_SIZE={TP_SIZE}, VOCAB_SIZE={VOCAB_SIZE}, EMBEDDING_DIM={EMBEDDING_DIM}")
    logger.info(f"词汇表: {VOCAB}")
    logger.info(f"输入 Token: {text_input}")
    logger.info(f"输入 Token IDs: {input_ids.flatten().tolist()}")
    logger.info("="*60)

    # 手动模拟每个 Rank 的执行过程
    partial_outputs = []
    all_embed_for_verification = []
    for tp_rank in range(TP_SIZE):
        logger.info(f"{'#' * 20} 模拟 Rank {tp_rank} 的计算过程 {'#' * 20}")
        # 词嵌入层
        embedding_layer = VocabParallelEmbedding_test(
            num_embed=VOCAB_SIZE,
            embed_dim=EMBEDDING_DIM,
            tp_rank=tp_rank,
            tp_size=TP_SIZE,
        )
        partition_size = VOCAB_SIZE // TP_SIZE
        # 打印当前 GPU 负责的词汇
        start_idx = embedding_layer.vocab_start_idx
        end_idx = embedding_layer.vocab_end_idx
        vocab_slice = [f"'{id_to_word[i]}':{i}" for i in range(start_idx, end_idx)]
        logger.info(f"Rank {tp_rank} 负责的词汇(及其全局 ID): {vocab_slice}")
        
        # 设置权重
        weights_data = torch.arange(start_idx, end_idx, dtype=torch.float32).unsqueeze(1) * 10.0 + \
                       torch.arange(EMBEDDING_DIM, dtype=torch.float32)
        with torch.no_grad():
            embedding_layer.weight.copy_(weights_data)
            all_embed_for_verification.append(weights_data)
        logger.info(f"Rank {tp_rank} 的权重分片: \n{embedding_layer.weight}")

        # 前向传播
        partial_y = embedding_layer(input_ids)
        partial_outputs.append(partial_y)
    
    # --- 模拟 All-Reduce 操作 ---
    logger.info(f"{'='*25} 模拟 All-Reduce 聚合 {'='*25}")
    for i, p_out in enumerate(partial_outputs):
        logger.info(f"来自 Rank {i} 的贡献: {p_out}")

    final_output = torch.stack(partial_outputs).sum(dim=0)
    logger.info(f"聚合后的最终结果: \n{final_output}")

    # --- 验证结果 ---
    logger.info(f"{'='*28} 验证结果 {'='*28}")
    full_weight_matrix = torch.cat(all_embed_for_verification, dim=0)
    logger.info(f"完整权重矩阵 (每一行代表一个单词的向量): \n{full_weight_matrix}")

    standard_embedding_layer = nn.Embedding(VOCAB_SIZE, EMBEDDING_DIM)
    with torch.no_grad():
        standard_embedding_layer.weight.copy_(full_weight_matrix)
    standard_output = standard_embedding_layer(input_ids)
    logger.info(f"标准 nn.Embedding 计算结果: \n{standard_output}")

    # 更加直观地对比单个词的结果
    word_to_check = "cat"
    word_id = word_to_id[word_to_check]
    word_index_in_batch = text_input.index(word_to_check)

    logger.info(f"--- 单独验证单词 '{word_to_check}' (ID: {word_id}) 的向量 ---")
    logger.info(f"并行计算得到的向量:\n{final_output[0, word_index_in_batch]}")
    logger.info(f"标准计算得到的向量:\n{standard_output[0, word_index_in_batch]}")

    are_equal = torch.allclose(final_output, standard_output)
    logger.info(f"并行计算结果与标准计算结果是否一致: {are_equal}")

if __name__ == "__main__":
    main()
