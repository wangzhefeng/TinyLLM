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
     # input example(after tokenization: token_ids)
    input_token_ids = torch.tensor([2, 3, 5, 1])
    logger.info(f"input_token_ids: {input_token_ids}, input_token_ids.shape: {input_token_ids.shape}")

    # vocabulary size
    vocab_size = max(input_token_ids) + 1
    logger.info(f"vocab_size: {vocab_size}")
    
    # embedding size
    output_dim = 3
    logger.info(f"output_dim: {output_dim}")
    
    # embedding layer
    torch.manual_seed(123)
    embedding_layer = torch.nn.Embedding(
        num_embeddings=vocab_size, 
        embedding_dim=output_dim
    )
    logger.info(f"embedding_layer.weight: \n{embedding_layer.weight}")
    logger.info(f"embedding_layer.weight.shape: {embedding_layer.weight.shape}")
    
    # input token_ids embeddings
    input_token_ids_embeddings = embedding_layer(input_token_ids)
    logger.info(f"input_token_ids_embeddings: \n{input_token_ids_embeddings}")


def random_data_embedding_v2():
     # input example(after tokenization: token_ids)
    input_token_ids = torch.tensor([2, 3, 5, 1])
    logger.info(f"input_token_ids: {input_token_ids}, input_token_ids.shape: {input_token_ids.shape}") 

    # vocabulary size
    vocab_size = max(input_token_ids) + 1
    logger.info(f"vocab_size: {vocab_size}")
    
    # embedding size
    output_dim = 3
    logger.info(f"output_dim: {output_dim}") 

    # embedding layer
    torch.manual_seed(123)
    embedding_layer = torch.nn.Embedding(
        num_embeddings=vocab_size, 
        embedding_dim=output_dim
    )
    logger.info(f"embedding_layer.weight: \n{embedding_layer.weight}")
    logger.info(f"embedding_layer.weight.shape: {embedding_layer.weight.shape}")

    # input token_ids embeddings
    input_token_ids_embeddings = embedding_layer(input_token_ids)
    logger.info(f"input_token_ids_embeddings: \n{input_token_ids_embeddings}")

    # one-hot encoded representation
    one_hot = torch.nn.functional.one_hot(input_token_ids).float()
    logger.info(f"one_hot: \n{one_hot}")
    # linear layer
    torch.manual_seed(123)
    linear_layer = torch.nn.Linear(
        in_features=vocab_size,
        out_features=output_dim,
        bias=False,
    )
    logger.info(f"linear_layer.weight: \n{linear_layer.weight}")
    logger.info(f"linear_layer.weight.shape: {linear_layer.weight.shape}")
    # linear_layer with embedding_layer weight
    linear_layer.weight = torch.nn.Parameter(embedding_layer.weight.T)
    logger.info(f"linear_layer.weight: \n{linear_layer.weight}")
    logger.info(f"linear_layer.weight.shape: {linear_layer.weight.shape}")
    # linear_layer with one-hot encoded representation
    one_hot_linear = linear_layer(one_hot)
    logger.info(f"one_hot_linear: \n{one_hot_linear}")

    one_hot_linear = one_hot @ linear_layer.weight.T
    logger.info(f"one_hot_linear: \n{one_hot_linear}")


def text_data_embedding():
    # input text
    input_strings = "Your journey starts with one step."
    logger.info(f"input_strings: {input_strings}")
    logger.info(f"input_strings length: {len(input_strings)}")
    
    # tokenizer
    import tiktoken
    tokenizer = tiktoken.get_encoding("gpt2")

    # input text tokenization
    input_token_ids = tokenizer.encode(text=input_strings, allowed_special={"<|endoftext|>"})
    input_token_ids = torch.tensor(input_token_ids)
    logger.info(f"input_token_ids: {input_token_ids}")
    logger.info(f"input_token_ids.shape: {input_token_ids.shape}")
    
    # embedding
    vocab_size = 50257
    output_dim = 3
    embedding_layer = torch.nn.Embedding(
        num_embeddings=vocab_size,
        embedding_dim=output_dim,
    )
    input_embedding = embedding_layer(input_token_ids)
    logger.info(f"input_embedding: \n{input_embedding}")
    logger.info(f"input_embedding.shape: {input_embedding.shape}")
    
    # single token_id
    input_token_id_0 = input_token_ids[0]
    logger.info(f"input_token_id_0: {input_token_id_0}")
    input_embedding_0 = embedding_layer(input_token_id_0)
    logger.info(f"input_embedding_0: \n{input_embedding_0}")
    logger.info(f"input_embedding_0.shape: {input_embedding_0.shape}")


def real_data_embedding():
    # data path
    data_path = "./dataset/pretrain/gpt"
    data_file = "the-verdict.txt"

    # tokenizer
    import tiktoken
    tokenizer = tiktoken.get_encoding("gpt2")

    # dataloader
    from data_provider.data_loader import create_dataloader
    batch_size = 8
    max_len = 4  # 1024
    train_dataset, train_dataloader = create_dataloader(
        data_path = data_path, 
        data_file = data_file,
        flag = "train", 
        train_ratio = 0.8,
        tokenizer = tokenizer,
        batch_size = batch_size,
        max_len = max_len,
        stride = max_len,
        num_workers = 0,
    )
    data_iter = iter(train_dataloader)
    batch_inputs, batch_targets = next(data_iter)
    logger.info(f"batch_inputs: \n{batch_inputs}")
    logger.info(f"batch_size shape: {batch_inputs.shape}")
    logger.info(f"batch_targets: \n{batch_targets}")
    logger.info(f"batch_targets shape: {batch_targets.shape}")
    
    # token embedding
    vocab_size = 50257
    output_dim = 256
    token_embedding_layer = torch.nn.Embedding(
        num_embeddings=vocab_size, 
        embedding_dim=output_dim
    )
    batch_inputs_token_embeddings = token_embedding_layer(batch_inputs)
    logger.info(f"batch_inputs_token_embeddings: \n{batch_inputs_token_embeddings}")
    logger.info(f"batch_inputs_token_embeddings shape: {batch_inputs_token_embeddings.shape}")
    batch_targets_token_embeddings = token_embedding_layer(batch_targets)
    logger.info(f"batch_targets_token_embeddings: \n{batch_targets_token_embeddings}")
    logger.info(f"batch_targets_token_embeddings shape: {batch_targets_token_embeddings.shape}")

    # position embedding
    context_length = max_len
    pos_embedding_layer = torch.nn.Embedding(
        num_embeddings=context_length, 
        embedding_dim=output_dim
    )
    batch_inputs_pos_embeddings = pos_embedding_layer(torch.arange(context_length))
    logger.info(f"batch_inputs_pos_embeddings: \n{batch_inputs_pos_embeddings}")
    logger.info(f"batch_inputs_pos_embeddings shape: {batch_inputs_pos_embeddings.shape}")
    
    # input embedding
    batch_inputs_embeddings = batch_inputs_token_embeddings + batch_inputs_pos_embeddings
    logger.info(f"batch_inputs_embeddings: \n{batch_inputs_embeddings}")
    logger.info(f"batch_inputs_embeddings shape: {batch_inputs_embeddings.shape}")

    batch_targets_embeddings = batch_targets_token_embeddings + batch_inputs_pos_embeddings
    logger.info(f"batch_targets_embeddings: \n{batch_targets_embeddings}")
    logger.info(f"batch_targets_embeddings shape: {batch_targets_embeddings.shape}")




# 测试代码 main 函数
def main():
    logger.info(f"{'-'*20}")
    logger.info(f"random data embedding...")
    logger.info(f"{'-'*20}")
    random_data_embedding()

    logger.info(f"{'-'*20}")
    logger.info(f"random data embedding v2...")
    logger.info(f"{'-'*20}")
    random_data_embedding_v2()

    # logger.info(f"{'-'*20}")
    # logger.info(f"text data embedding...")
    # logger.info(f"{'-'*20}")
    # text_data_embedding()

    # logger.info(f"{'-'*20}")
    # logger.info(f"real data embedding...")
    # logger.info(f"{'-'*20}")
    # real_data_embedding()

if __name__ == "__main__":
    main()
