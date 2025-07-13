# -*- coding: utf-8 -*-

# ***************************************************
# * File        : embedding_test.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-02-10
# * Version     : 0.1.021021
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import os
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)


import torch

from utils.log_util import logger

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


def random_data_embedding():
     # input example(after tokenization: token ids)
    token_ids = torch.tensor([2, 3, 5, 1])
    logger.info(f"token_ids.shape: {token_ids.shape}")
    # vocabulary of 6 words
    vocab_size = 6  # max(input_ids) + 1
    # embedding size 3
    output_dim = 3
    # embedding layer
    torch.manual_seed(123)
    embedding_layer = torch.nn.Embedding(
        num_embeddings=vocab_size, 
        embedding_dim=output_dim
    )
    logger.info(f"embedding_layer.weight: \n{embedding_layer.weight}")
    logger.info(f"embedding_layer.weight.shape: {embedding_layer.weight.shape}")
    logger.info(f"embedding_layer(torch.tensor([3])): \n{embedding_layer(torch.tensor([3]))}")
    logger.info(f"embedding_layer(token_ids): \n{embedding_layer(token_ids)}")


def text_tokenization_embedding():
    # input text
    input_strings = "Your journey starts with one step."
    logger.info(f"input_strings: {input_strings}")
    logger.info(f"input_strings length: {len(input_strings)}")
    
    # tokenization
    import tiktoken
    tokenizer = tiktoken.get_encoding("gpt2")
    token_ids = tokenizer.encode(text=input_strings, allowed_special={"<|endoftext|>"})
    token_ids = torch.tensor(token_ids)
    logger.info(f"token_ids: {token_ids}")
    logger.info(f"token_ids.shape: {token_ids.shape}")
    
    # embedding
    vocab_size = 50257
    output_dim = 3
    embedding_layer = torch.nn.Embedding(
        num_embeddings=vocab_size, 
        embedding_dim=output_dim
    ) 
    embedding = embedding_layer(token_ids)
    logger.info(f"embedding: \n{embedding}")
    logger.info(f"embedding.shape: {embedding.shape}")
    
    # single token_id
    token_id = token_ids[0]
    logger.info(f"token_id: {token_id}")
    embedding = embedding_layer(token_id)
    logger.info(f"embedding: \n{embedding}")
    logger.info(f"embedding.shape: {embedding.shape}")


def real_data_embedding(): 
    # data download & load
    from data_provider.pretrain.data_load import data_load
    raw_text = data_load(
        url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"
    )

    # dataloader
    from data_provider.data_loader import create_dataloader
    batch_size = 8
    max_length = 4  # 1024
    dataloader = create_dataloader(
        raw_text,
        batch_size=batch_size,
        max_length=max_length,
        stride=max_length,
        shuffle=False,
        drop_last=True,
    )
    data_iter = iter(dataloader)
    token_ids, targets = next(data_iter)
    logger.info(f"Token IDs: \n{token_ids}")
    logger.info(f"Token IDs shape: {token_ids.shape}")
    logger.info(f"Targets: \n{targets}")
    logger.info(f"Targets shape: {targets.shape}")
    # token embedding
    # ---------------
    vocab_size = 50257
    output_dim = 256
    token_embedding_layer = torch.nn.Embedding(
        num_embeddings=vocab_size, 
        embedding_dim=output_dim
    )
    token_embeddings = token_embedding_layer(token_ids)
    logger.info(f"Token embeddings: \n{token_embeddings}")
    logger.info(f"Token embeddings shape: {token_embeddings.shape}")
    target_embeddings = token_embedding_layer(targets)
    logger.info(f"Target embeddings: \n{target_embeddings}")
    logger.info(f"Target embeddings shape: {target_embeddings.shape}")

    # position embedding
    # ---------------
    context_length = max_length
    pos_embedding_layer = torch.nn.Embedding(
        num_embeddings=context_length, 
        embedding_dim=output_dim
    )
    pos_embeddings = pos_embedding_layer(torch.arange(max_length))
    logger.info(f"Position embeddings: \n{pos_embeddings}")
    logger.info(f"Position embeddings shape: {pos_embeddings.shape}")
    
    # input embedding
    # ---------------
    input_embeddings = token_embeddings + pos_embeddings
    logger.info(f"Input embeddings: \n{input_embeddings}")
    logger.info(f"Input embeddings shape: {input_embeddings.shape}")




# 测试代码 main 函数
def main():
    # ------------------------------
    # random data embedding
    # ------------------------------
    random_data_embedding()
    
    # ------------------------------
    # real data embedding
    # ------------------------------
    text_tokenization_embedding()

    real_data_embedding()

if __name__ == "__main__":
    main()
