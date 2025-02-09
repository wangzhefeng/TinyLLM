# -*- coding: utf-8 -*-

# ***************************************************
# * File        : embedding.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-01-24
# * Version     : 1.0.012400
# * Description : https://github.com/rasbt/LLMs-from-scratch/blob/2dc46bedc6e86b79a16c4099e557564cd23e03ef/ch02/03_bonus_embedding-vs-matmul/embeddings-and-linear-layers.ipynb
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

from utils.log_util import logger

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


def token_embedding():
    pass


def position_embedding():
    pass


def embedding():
    pass




# 测试代码 main 函数
def main():
    import torch
    # ------------------------------
    # token embeddings
    # ------------------------------
    # input example(after tokenization)
    input_ids = torch.tensor([2, 3, 5, 1])
    # vocabulary of 6 words
    vocab_size = 6  # max(input_ids) + 1
    # embedding size 3
    output_dim = 3
    # embedding layer
    torch.manual_seed(123)
    embedding_layer = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=output_dim)
    logger.info(f"embedding_layer.weight: \n{embedding_layer.weight}")
    logger.info(f"embedding_layer(torch.tensor([3])): \n{embedding_layer(torch.tensor([3]))}")
    logger.info(f"embedding_layer(input_ids): \n{embedding_layer(input_ids)}")
    
    # ------------------------------
    # encoding word positions
    # ------------------------------
    from data_provider.data_load_pretrain import data_load
    from data_provider.data_loader import create_dataloader

    # params
    vocab_size = 50257
    output_dim = 256
    batch_size = 8
    max_length = 4  # 1024
    
    # data download & load
    raw_text = data_load(
        url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"
    )

    # dataloader 
    dataloader = create_dataloader(
        raw_text,
        batch_size=batch_size,
        max_length=max_length,
        stride=max_length,
        shuffle=False,
        drop_last=True,
    )
    data_iter = iter(dataloader)
    inputs, targets = next(data_iter)
    logger.info(f"Token IDs: \n{inputs}")
    logger.info(f"Inputs shape: {inputs.shape}")
    
    # token embedding
    token_embedding_layer = torch.nn.Embedding(
        num_embeddings=vocab_size, 
        embedding_dim=output_dim
    )
    token_embeddings = token_embedding_layer(inputs)
    logger.info(f"Token embeddings: \n{token_embeddings}")
    logger.info(f"Token embeddings shape: {token_embeddings.shape}")
    
    # position embedding
    context_length = max_length
    pos_embedding_layer = torch.nn.Embedding(
        num_embeddings=context_length, 
        embedding_dim=output_dim
    )
    pos_embeddings = pos_embedding_layer(torch.arange(max_length))
    logger.info(f"Position embeddings: \n{pos_embeddings}")
    logger.info(f"Position embeddings shape: {pos_embeddings.shape}")
    
    # input embedding
    input_embeddings = token_embeddings + pos_embeddings
    logger.info(f"Input embeddings: \n{input_embeddings}")
    logger.info(f"Input embeddings shape: {input_embeddings.shape}")
    
if __name__ == "__main__":
    main()
