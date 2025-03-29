# -*- coding: utf-8 -*-

# ***************************************************
# * File        : attention_test.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2025-02-10
# * Version     : 0.1.021021
# * Description : https://github.com/rasbt/LLMs-from-scratch/blob/main/ch03/02_bonus_efficient-multihead-attention/mha-implementations.ipynb
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import os
import sys
ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
from importlib.metadata import version

import torch

from layers.attention import (
    CasualAttention, 
    MultiHeadAttentionWrapper, 
    MultiHeadAttention, 
    MultiHeadAttentionCombinedQKV, 
    MHAPyTorchClass, 
    MHAEinsum, 
    MHAPyTorchFlexAttention, 
    MHAPyTorchScaledDotProduct, 
    MHAPyTorchSDPAWithoutFlash, 
    current_version, 
    required_version
)
from utils.device import device_setting
from utils.log_util import logger

logger.info(f"torch version: {version('torch')}")

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


# device
device = device_setting()


def random_data_attention():
    # tokens embeddings
    embeddings = torch.tensor(
        [[0.43, 0.15, 0.89], # Your     (x^1)
        [0.55, 0.87, 0.66], # journey  (x^2)
        [0.57, 0.85, 0.64], # starts   (x^3)
        [0.22, 0.58, 0.33], # with     (x^4)
        [0.77, 0.25, 0.10], # one      (x^5)
        [0.05, 0.80, 0.55]] # step     (x^6)
    )
    logger.info(f"embeddings: \n{embeddings}")
    logger.info(f"embeddings.shape: {embeddings.shape}")

    # batch token embedding
    batch = torch.stack((embeddings, embeddings), dim=0)
    logger.info(f"batch: \n{batch}")
    logger.info(f"batch.shape: {batch.shape}")
    
    # params
    batch_size, context_len, embed_dim = batch.shape
    d_out = 2
    torch.manual_seed(123)
    # ------------------------------
    # method 1: CausalAttention MHA wrapper
    # ------------------------------
    # attention
    casual_attn = CasualAttention(
        d_in=embed_dim, 
        d_out=d_out, 
        context_length=context_len, 
        dropout=0.0,
    )
    context_vecs = casual_attn(batch)
    logger.info(f"context_vecs: \n{context_vecs}")
    logger.info(f"context_vecs.shape: {context_vecs.shape}")
    
    # multi-head attention
    mha = MultiHeadAttentionWrapper(
        d_in=embed_dim, 
        d_out=d_out, 
        context_length=context_len, 
        dropout=0.0, 
        num_heads=2,
    )
    context_vecs = mha(batch)
    logger.info(f"context_vecs: \n{context_vecs}")
    logger.info(f"context_vecs.shape: {context_vecs.shape}")




# 测试代码 main 函数
def main():
    # ------------------------------
    # random data 1
    # ------------------------------
    random_data_attention()
    # ------------------------------
    # random data 2
    # ------------------------------
    # params
    batch_size = 8
    context_len = 1024
    embed_dim = 768
    
    # embedding
    embeddings = torch.randn((batch_size, context_len, embed_dim), device=device)
    logger.info(f"embeddings: \n{embeddings}")
    logger.info(f"embeddings.shape: {embeddings.shape}")
    # ------------------------------
    # method 1: CausalAttention MHA wrapper
    # ------------------------------
    mha = MultiHeadAttentionWrapper(
        d_in=embed_dim,
        d_out=embed_dim // 12,
        context_length=context_len,
        dropout=0.0,
        num_heads=12,
        qkv_bias=False,
    ).to(device)
    context_vecs = mha(embeddings)
    logger.info(f"context_vecs: \n{context_vecs}")
    logger.info(f"context_vecs.shape: {context_vecs.shape}")
 
    # ------------------------------
    # method 2: The multi-head attention
    # ------------------------------
    mha = MultiHeadAttention(
        d_in=embed_dim, 
        d_out=embed_dim, 
        context_len=context_len, 
        dropout=0.0, 
        num_heads=12,
        qkv_bias=False
    ).to(device)
    context_vecs = mha(embeddings)
    logger.info(f"context_vecs: \n{context_vecs}")
    logger.info(f"context_vecs.shape: \n{context_vecs.shape}")
    
    # ------------------------------
    # method 3: An alternative multi-head attention with combined weights
    # ------------------------------  
    mha_combined_qkv = MultiHeadAttentionCombinedQKV(
        d_in=embed_dim,
        d_out=embed_dim,
        context_length=context_len,
        dropout=0.0,
        num_heads=12,
        qkv_bias=False
    ).to(device)

    out = mha_combined_qkv(embeddings)
    print(out.shape)
    
    # ------------------------------
    # Multi-head attention with Einsum
    # ------------------------------
    mha_einsum = MHAEinsum(
        d_in=embed_dim,
        d_out=embed_dim,
        context_length=context_len,
        dropout=0.0,
        num_heads=12,
        qkv_bias=False
    ).to(device)

    out = mha_einsum(embeddings)
    print(out.shape)
    
    # ------------------------------
    # Multi-head attention with PyTorch's scaled dot product attention and FlashAttention
    # ------------------------------
    mha_pytorch_scaled = MHAPyTorchScaledDotProduct(
        d_in=embed_dim,
        d_out=embed_dim,
        context_length=context_len,
        dropout=0.0,
        num_heads=12,
        qkv_bias=False
    ).to(device)

    out = mha_pytorch_scaled(embeddings)
    print(out.shape)
    
    # ------------------------------
    # PyTorch's scaled dot product attention without FlashAttention
    # ------------------------------
    mha_pytorch_sdpa_no_flash = MHAPyTorchSDPAWithoutFlash(
        d_in=embed_dim,
        d_out=embed_dim,
        context_length=context_len,
        dropout=0.0,
        num_heads=12,
        qkv_bias=False
    ).to(device)

    out = mha_pytorch_sdpa_no_flash(embeddings)
    print(out.shape)
    
    # ------------------------------
    # Using PyTorch's torch.nn.MultiheadAttention
    # ------------------------------
    mha_pytorch_class_default = MHAPyTorchClass(
        d_in=embed_dim,
        d_out=embed_dim,
        context_length=context_len,
        dropout=0.0,
        num_heads=12,
        qkv_bias=False
    ).to(device)

    out = mha_pytorch_class_default(embeddings)
    print(out.shape)
    
    # ------------------------------
    # Using PyTorch's torch.nn.MultiheadAttention with scaled_dot_product_attention
    # ------------------------------
    mha_pytorch_class_noweights = MHAPyTorchClass(
        d_in=embed_dim,
        d_out=embed_dim,
        context_length=context_len,
        dropout=0.0,
        num_heads=12,
        qkv_bias=False,
        need_weights=False # NEW!
    ).to(device)

    out = mha_pytorch_class_noweights(embeddings)
    print(out.shape)
    
    # ------------------------------
    # Using PyTorch's FlexAttention
    # ------------------------------
    if current_version >= required_version and torch.cuda.is_available():
        mha_pytorch_flex = MHAPyTorchFlexAttention(
            d_in=embed_dim,
            d_out=embed_dim,
            context_length=context_len,
            dropout=0.0,
            num_heads=12,
            qkv_bias=False
        ).to(device)

        out = mha_pytorch_flex(embeddings)
        print(out.shape)

if __name__ == "__main__":
   main()
