# -*- coding: utf-8 -*-

# ***************************************************
# * File        : gpt_test.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2025-02-12
# * Version     : 0.1.021223
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import os
import sys
ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import torch

from utils.log_util import logger

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]





# 测试代码 main 函数
def main():
    # model params
    GPT_CONFIG_124M = {
        "vocab_size": 50257,     # Vocabular size
        "context_length": 1024,  # Context length
        "emb_dim": 768,          # Embedding dimension
        "n_heads": 12,           # Number of attention heads
        "n_layers": 12,          # Number of transformer layers
        "dropout": 0.1,          # Dropout rate
        "qkv_bias": False,       # Query-Key-Value bias
    }
    
    # tokenizer
    import tiktoken
    tokenizer = tiktoken.get_encoding("gpt2")
    
    # input data
    batch = []
    text1 = "Every effort moves you"
    text2 = "Every day hold a"
    text3 = "Hello, I am"
    batch.append(torch.tensor(tokenizer.encode(text1)).unsqueeze(0))
    batch.append(torch.tensor(tokenizer.encode(text2)).unsqueeze(0))
    batch.append(torch.tensor(tokenizer.encode(text3)).unsqueeze(0))
    batch = torch.stack(batch, dim=0)
    logger.info(f"batch: \n{batch}")
    logger.info(f"batch.shape: {batch.shape}")
    
    """
    # ------------------------------
    # GPT model
    # ------------------------------
    torch.manual_seed(123)
    # model
    model = GPT(GPT_CONFIG_124M)
    
    # model forward
    logits = model(batch)
    logger.info(f"Input: \n{batch}")
    logger.info(f"Output: \n{logits}")
    logger.info(f"Output shape: {logits.shape}")
    # model params
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total number of parameters: {total_params:,}")
    # logger.info(f"Token embedding layer shape: {model.tok_emb.weight.shape}")
    # logger.info(f"Output layer shape: {model.out_head.weight.shape}")
    total_params_gpt2 = total_params - sum(p.numel() for p in model.out_head.parameters())
    logger.info(f"Number of trainable parameters considering weight tying: {total_params_gpt2:,}")
    # compute memory demand of the model
    total_size_bytes = total_params * 4  # total size in bytes(assuming float32, 4 bytes per parameter)
    # convert to megabytes
    total_size_mb = total_size_bytes / (1024 * 1024)
    logger.info(f"Total size of the model: {total_size_mb:.2f} MB")

    # ------------------------------
    # generating text: v1
    # ------------------------------
    start_context = "Hello, I am"
    encoded = tokenizer.encode(start_context)
    logger.info(f"encoded: {encoded}")
    
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    logger.info(f"encoded_tensor.shape: {encoded_tensor.shape}")

    # disable dropout
    model.eval()

    out = generate_text_simple(
        model = model,
        idx = encoded_tensor,
        max_new_tokens = 6,
        context_size=GPT_CONFIG_124M["context_length"],
    )
    logger.info(f"Output: {out}")
    logger.info(f"Output length: {len(out[0])}") 
    
    decoded_text = tokenizer.decode(out.squeeze(0).tolist())
    logger.info(decoded_text)
    
    # ------------------------------
    # generating text: v2
    # ------------------------------
    token_ids = generate(
        model = model,
        idx = text_to_token_ids("Every effort moves you", tokenizer),
        max_new_toknes = 15,
        context_size = GPT_CONFIG_124M["context_length"],
        top_k = 25,
        temperature = 1.4,
    )
    logger.info(f"Output text: \n{token_ids_to_text(token_ids, tokenizer)}")
    """


if __name__ == "__main__":
    main()
