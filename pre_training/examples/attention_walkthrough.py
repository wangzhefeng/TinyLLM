# -*- coding: utf-8 -*-

# ***************************************************
# * File        : simple_attention.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-01-25
# * Version     : 0.1.012501
# * Description : https://github.com/rasbt/LLMs-from-scratch/blob/main/ch03/02_bonus_efficient-multihead-attention/mha-implementations.ipynb
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
import torch.nn as nn

from layers.attention import (
    CasualAttention, MultiHeadAttentionWrapper, 
    MultiHeadAttention, 
    MultiHeadAttentionCombinedQKV, 
    MHAEinsum, 
    MHAPyTorchScaledDotProduct, 
    MHAPyTorchSDPAWithoutFlash, 
    MHAPyTorchClass, 
    MHAPyTorchFlexAttention, 
    MultiHeadAttentionRoPE, 
    GroupedQueryAttention, 
)
from utils.device import device_setting
from utils.log_util import logger

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


# ------------------------------
# attention test
# ------------------------------
def atttention_simple_test(inputs_embed):
    # ------------------------------
    # Step 1: attention scores w
    # attention scores(query x(2) with other x(i), i=1,...,6)
    # ------------------------------
    logger.info("-" * 40)
    logger.info(f"Calculate Attention Scores...")
    logger.info("-" * 40)

    query_2 = inputs_embed[1, :]
    attn_scores_2 = torch.empty(inputs_embed.shape[0])
    for i, x_i in enumerate(inputs_embed):
        logger.info(f"x_{i}: {x_i}")
        logger.info(f"query_2: {query_2}")
        attn_scores_2[i] = torch.dot(x_i, query_2)
        logger.info(f"attn_scores_2[{i}]: {attn_scores_2[i]}")
        logger.info("")

    logger.info(f"attn_scores_2: {attn_scores_2}")
    logger.info(f"attn_scores_2.shape: {attn_scores_2.shape}")
    # ------------------------------
    # Step 2: attention weights
    # ------------------------------
    logger.info("-" * 40)
    logger.info(f"Calculate Attention Weights...")
    logger.info("-" * 40)

    attn_weights_2_tmp = attn_scores_2 / attn_scores_2.sum()
    logger.info(f"attn_weights_2_tmp: {attn_weights_2_tmp}")
    # or
    def softmax_naive(x):
        return torch.exp(x) / torch.sum(torch.exp(x))
    attn_weights_2_tmp_softmax = softmax_naive(attn_scores_2)
    logger.info(f"attn_weights_2_tmp_softmax: {attn_weights_2_tmp_softmax}")
    # or
    attn_weights_2_tmp_softmax = torch.softmax(attn_scores_2, dim=0)
    logger.info(f"attn_weights_2_tmp_softmax: {attn_weights_2_tmp_softmax}")
    # ------------------------------
    # Step 3: context vector
    # ------------------------------
    logger.info("-" * 40)
    logger.info(f"Calculate Context Vector...")
    logger.info("-" * 40)

    context_vec_2 = torch.zeros(query_2.shape)
    for i, x_i in enumerate(inputs_embed):
        logger.info(f"attn_weights_2_tmp[i]: {attn_weights_2_tmp[i]}")
        logger.info(f"x_i: {x_i}")
        logger.info(f"attn_weights_2_tmp[i] * x_i: {attn_weights_2_tmp[i] * x_i}")
        context_vec_2 += attn_weights_2_tmp[i] * x_i
        logger.info(f"context_vec_2: {context_vec_2}")
        logger.info("")

    logger.info(f"context_vec_2: {context_vec_2}")
    # ------------------------------
    # all context vectors
    # ------------------------------
    logger.info("-" * 40)
    logger.info(f"Calculate All Context Vector...")
    logger.info("-" * 40)

    # attention scores
    attn_scores = torch.empty(inputs_embed.shape[0], inputs_embed.shape[0])
    for i, x_i in enumerate(inputs_embed):
        for j, x_j in enumerate(inputs_embed):
            attn_scores[i, j] = torch.dot(x_i, x_j)
    logger.info(f"attn_scores: \n{attn_scores}")
    # or
    attn_scores = inputs_embed @ inputs_embed.T  # (6, 6)
    logger.info(f"attn_scores: \n{attn_scores}")

    # attention weights
    attn_weights = torch.softmax(attn_scores, dim = -1)  # (6, 6)
    logger.info(f"attn_weights: \n{attn_weights}")

    # context vectors
    context_vecs = attn_weights @ inputs_embed  # (6, 6) @ (6, 3) -> (6, 3)
    logger.info(f"context_vecs: \n{context_vecs}")


# ------------------------------
# self-attention test
# ------------------------------
def self_attention_simple_test(inputs_embed):
    # ------------------------------
    # self-attention with trainable weights
    # ------------------------------
    logger.info("-" * 40)
    logger.info(f"Calculate Self-Attention with trainable weights...")
    logger.info("-" * 40)

    # params
    d_model = inputs_embed.shape[1]  # the input embedding size, d=3
    d_out = 2  # the output embedding size, d=2
    logger.info(f"d_model: {d_model}")
    logger.info(f"d_out: {d_out}")

    # attention weights
    torch.manual_seed(123)
    W_query = torch.nn.Parameter(torch.rand(d_model, d_out), requires_grad=False)
    W_key = torch.nn.Parameter(torch.rand(d_model, d_out), requires_grad=False)
    W_value = torch.nn.Parameter(torch.rand(d_model, d_out), requires_grad=False)
    logger.info(f"W_query: \n{W_query}")
    logger.info(f"W_key: \n{W_key}")
    logger.info(f"W_value: \n{W_value}")


    # query: x2
    x_2 = inputs_embed[1]  # second input element
    logger.info(f"x_2: {x_2}")
    # query, key, value vectors
    query_2 = x_2 @ W_query
    key_2 = x_2 @ W_key
    value_2 = x_2 @ W_value
    logger.info(f"query_2: {query_2}")
    logger.info(f"key_2: {key_2}")
    logger.info(f"value_2: {value_2}")


    # query: input_embed
    logger.info(f"inputs_embed: \n{inputs_embed}")
    # query, key, value vectors
    queries = inputs_embed @ W_query
    keys = inputs_embed @ W_key
    values = inputs_embed @ W_value
    logger.info(f"queries: \n{queries} \nqueries.shape: {queries.shape}")
    logger.info(f"keys: \n{keys} \nkeys.shape: {keys.shape}")
    logger.info(f"values: \n{values} \nvalues.shape: {values.shape}")


    # unnormalized attention scores: w2j
    attn_scores_21 = query_2 @ keys[0]
    attn_scores_22 = query_2 @ keys[1]
    attn_scores_23 = query_2 @ keys[2]
    attn_scores_24 = query_2 @ keys[3]
    attn_scores_25 = query_2 @ keys[4]
    attn_scores_26 = query_2 @ keys[5]
    logger.info(f"attn_scores_21: {attn_scores_21}")
    logger.info(f"attn_scores_22: {attn_scores_22}")
    logger.info(f"attn_scores_23: {attn_scores_23}")
    logger.info(f"attn_scores_24: {attn_scores_24}")
    logger.info(f"attn_scores_25: {attn_scores_25}")
    logger.info(f"attn_scores_26: {attn_scores_26}")


    # unnormalized attention scores
    attn_scores_1 = queries[0] @ keys.T
    attn_scores_2 = query_2    @ keys.T
    attn_scores_3 = queries[2] @ keys.T
    attn_scores_4 = queries[3] @ keys.T
    attn_scores_5 = queries[4] @ keys.T
    attn_scores_6 = queries[5] @ keys.T
    logger.info(f"attn_scores_1: {attn_scores_1}")
    logger.info(f"attn_scores_2: {attn_scores_2}")
    logger.info(f"attn_scores_3: {attn_scores_3}")
    logger.info(f"attn_scores_4: {attn_scores_4}")
    logger.info(f"attn_scores_5: {attn_scores_5}")
    logger.info(f"attn_scores_6: {attn_scores_6}")


    # attention weights
    d_k = keys.shape[1]
    attn_weights_1 = torch.softmax(attn_scores_1 / d_k ** 0.5, dim=-1)
    attn_weights_2 = torch.softmax(attn_scores_2 / d_k ** 0.5, dim=-1)
    attn_weights_3 = torch.softmax(attn_scores_3 / d_k ** 0.5, dim=-1)
    attn_weights_4 = torch.softmax(attn_scores_4 / d_k ** 0.5, dim=-1)
    attn_weights_5 = torch.softmax(attn_scores_5 / d_k ** 0.5, dim=-1)
    attn_weights_6 = torch.softmax(attn_scores_6 / d_k ** 0.5, dim=-1)
    logger.info(f"attn_weights_1: {attn_weights_1}")
    logger.info(f"attn_weights_2: {attn_weights_2}")
    logger.info(f"attn_weights_3: {attn_weights_3}")
    logger.info(f"attn_weights_4: {attn_weights_4}")
    logger.info(f"attn_weights_5: {attn_weights_5}")
    logger.info(f"attn_weights_6: {attn_weights_6}")


    # context vector for input query vector 2
    context_vec_1 = attn_weights_1 @ values
    context_vec_2 = attn_weights_2 @ values
    context_vec_3 = attn_weights_3 @ values
    context_vec_4 = attn_weights_4 @ values
    context_vec_5 = attn_weights_5 @ values
    context_vec_6 = attn_weights_6 @ values
    logger.info(f"context_vec_1: {context_vec_1}")
    logger.info(f"context_vec_2: {context_vec_2}")
    logger.info(f"context_vec_3: {context_vec_3}")
    logger.info(f"context_vec_4: {context_vec_4}")
    logger.info(f"context_vec_5: {context_vec_5}")
    logger.info(f"context_vec_6: {context_vec_6}")


# ------------------------------
# self-attention class v1
# ------------------------------
class SelfAttention_V1(nn.Module):
    
    def __init__(self, d_model, d_out):
        super().__init__()

        self.W_query = nn.Parameter(torch.rand(d_model, d_out), requires_grad=True)
        self.W_key = nn.Parameter(torch.rand(d_model, d_out), requires_grad=True)
        self.W_value = nn.Parameter(torch.rand(d_model, d_out), requires_grad=True)

    def forward(self, x):
        queries = x @ self.W_query
        keys = x @ self.W_key
        values = x @ self.W_value

        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        context_vec = attn_weights @ values

        return context_vec


# ------------------------------
# self-attention class v2
# ------------------------------
class SelfAttention_V2(nn.Module):
    
    def __init__(self, d_model, d_out, qkv_bias=False):
        super().__init__()

        self.W_query = nn.Linear(d_model, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_model, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_model, d_out, bias=qkv_bias)
    
    def forward(self, x):
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)
    
        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        context_vec = attn_weights @ values
        
        return context_vec


# ------------------------------
# Causal Self-attention test
# ------------------------------
def causal_self_attention_test(inputs_embed):
    torch.manual_seed(123)
    d_model = inputs_embed.shape[1]  # the input embedding size, d=3
    d_out = 2  # the output embedding size, d=2
    sa_v1 = SelfAttention_V1(d_model, d_out)
    # ------------------------------
    # causal attention mask
    # ------------------------------
    # simple method
    # --------------
    # attention weights
    queries = inputs_embed @ sa_v1.W_query
    keys = inputs_embed @ sa_v1.W_key
    values = inputs_embed @ sa_v1.W_value
    attn_scores = queries @ keys.T
    logger.info(f"attn_scores: \n{attn_scores}")
    attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
    logger.info(f"attn_weights: \n{attn_weights}")
    # masked attention weights
    context_length = attn_scores.shape[0]
    logger.info(f"context_length: {context_length}")
    mask = torch.tril(torch.ones(context_length, context_length), diagonal=0)
    logger.info(f"mask: \n{mask}")
    masked_attn_weights = attn_weights * mask
    logger.info(f"masked_attn_weights: \n{masked_attn_weights}")
    # normalization masked attention weights
    row_sums = masked_attn_weights.sum(dim=-1, keepdim=True)
    masked_attn_weights_norm = masked_attn_weights / row_sums
    logger.info(f"masked_attn_weights_norm: \n{masked_attn_weights_norm}")
    
    # efficient approach
    # --------------
    # attention weights
    queries = inputs_embed @ sa_v1.W_query
    keys = inputs_embed @ sa_v1.W_key
    values = inputs_embed @ sa_v1.W_value
    attn_scores = queries @ keys.T
    logger.info(f"attn_scores: \n{attn_scores}")
    # masked attention scores
    context_length = attn_scores.shape[0]
    logger.info(f"context_length: {context_length}")
    mask = torch.triu(torch.ones(context_length, context_length), diagonal=1).bool()
    logger.info(f"mask: \n{mask}")
    masked_attn_scores = attn_scores.masked_fill(mask, -torch.inf)
    logger.info(f"masked_attn_scores: \n{masked_attn_scores}")
    # normalization masked attention weights
    masked_attn_weights_norm = torch.softmax(masked_attn_scores / keys.shape[-1] ** 0.5, dim=-1)
    logger.info(f"masked_attn_weights_norm: \n{masked_attn_weights_norm}")
    # attention mask with dropout
    torch.manual_seed(123)
    dropout = torch.nn.Dropout(0.5)
    masked_attn_weights_norm_dropout = dropout(masked_attn_weights_norm)
    logger.info(f"masked_attn_weights_norm_dropout: \n{masked_attn_weights_norm_dropout}")
    
    # ------------------------------
    # context vector
    # ------------------------------
    context_vec = masked_attn_weights_norm @ values
    logger.info(f"context_vec: \n{context_vec}")


# ------------------------------
# Causal Multi-Head Self-attention
# ------------------------------
def causal_self_attention_class_test(inputs_embed):
    # batch token embedding
    batch = torch.stack((inputs_embed, inputs_embed), dim=0)
    logger.info(f"batch: \n{batch}")
    logger.info(f"batch.shape: {batch.shape}")
    
    # params
    batch_size, context_len, embed_dim = batch.shape
    d_out = 2
    
    # attention
    torch.manual_seed(123)
    casual_attn = CasualAttention(
        d_model=embed_dim, 
        d_out=d_out, 
        context_length=context_len, 
        dropout=0.0,
    )
    context_vecs = casual_attn(batch)
    logger.info(f"context_vecs: \n{context_vecs}")
    logger.info(f"context_vecs.shape: {context_vecs.shape}") 


# ------------------------------
# Causal Multi-Head Self-attention
# ------------------------------
def causal_multi_head_self_attention_class_test(inputs_embed):
    # batch token embedding
    batch = torch.stack((inputs_embed, inputs_embed), dim=0)
    logger.info(f"batch: \n{batch}")
    logger.info(f"batch.shape: {batch.shape}")
    
    # params
    batch_size, context_len, embed_dim = batch.shape
    d_out = 2

    # multi-head attention
    torch.manual_seed(123)
    mha = MultiHeadAttentionWrapper(
        d_model=embed_dim, 
        d_out=d_out, 
        context_length=context_len, 
        n_heads=2,
        dropout=0.0, 
    )
    context_vecs = mha(batch)
    logger.info(f"context_vecs: \n{context_vecs}")
    logger.info(f"context_vecs.shape: {context_vecs.shape}")




# 测试代码 main 函数
def main():
    # ------------------------------
    # device
    # ------------------------------
    device = device_setting(verbose=True)
    
    # ------------------------------
    # input text -> tokenization -> token IDs embedding
    # ------------------------------
    logger.info("-" * 40)
    logger.info(f"Input...")
    logger.info("-" * 40)
    # input text
    inputs = "Your journey starts with one step."
    logger.info(f"inputs: {inputs}")

    # token ids
    # ...

    # token ids embeddings
    inputs_embed = torch.tensor(
        [[0.43, 0.15, 0.89],  # Your     (x^1)
        [0.55, 0.87, 0.66],   # journey  (x^2)
        [0.57, 0.85, 0.64],   # starts   (x^3)
        [0.22, 0.58, 0.33],   # with     (x^4)
        [0.77, 0.25, 0.10],   # one      (x^5)
        [0.05, 0.80, 0.55]]   # step     (x^6)
    )
    logger.info(f"inputs_embed: \n{inputs_embed}")
    logger.info(f"inputs_embed.shape: {inputs_embed.shape}")
    # ------------------------------
    # attention test
    # ------------------------------
    atttention_simple_test(inputs_embed)

    # ------------------------------
    # self-attention test
    # ------------------------------
    self_attention_simple_test(inputs_embed)

    # ------------------------------
    # self-attention class v1
    # ------------------------------
    torch.manual_seed(123)
    d_model = inputs_embed.shape[1]  # the input embedding size, d=3
    d_out = 2  # the output embedding size, d=2
    sa_v1 = SelfAttention_V1(d_model, d_out)
    sa_v1_output = sa_v1(inputs_embed)
    logger.info(f"sa_v1_output: \n{sa_v1_output}")
    
    # ------------------------------
    # self-attention class v2
    # ------------------------------
    torch.manual_seed(789)
    d_model = inputs_embed.shape[1]  # the input embedding size, d=3
    d_out = 2  # the output embedding size, d=2
    sa_v2 = SelfAttention_V2(d_model, d_out)
    sa_v2_output = sa_v2(inputs_embed)
    logger.info(f"sa_v2_output: \n{sa_v2_output}")

    # ------------------------------
    # Causal Self-attention test
    # ------------------------------
    causal_self_attention_test(inputs_embed)

    # ------------------------------
    # Causal Multi-Head Self-attention
    # ------------------------------
    causal_self_attention_class_test(inputs_embed)
    
    # ------------------------------
    # Causal Multi-Head Self-attention
    # ------------------------------
    causal_multi_head_self_attention_class_test(inputs_embed)
    
    # ------------------------------
    # Multi-Head Self-attention test input
    # ------------------------------
    # params
    batch_size = 8
    context_len = 1024
    embed_dim = 768
    n_heads = 12
    llama3_context_len = 8129
    llama3_theta_base = 500_000
    
    # input embedding
    embeddings = torch.randn((batch_size, context_len, embed_dim), device=device)
    # logger.info(f"embeddings: \n{embeddings}")
    logger.info(f"embeddings.shape: {embeddings.shape}")
    # ------------------------------
    # CausalAttention MHA wrapper
    # ------------------------------
    mha_wrapper = MultiHeadAttentionWrapper(
        d_model=embed_dim,
        d_out=embed_dim,
        n_heads=n_heads,
        context_length=context_len,
        dropout=0.0,
        qkv_bias=False,
        proj=True
    ).to(device)
    context_vecs = mha_wrapper(embeddings)
    # logger.info(f"context_vecs: \n{context_vecs}")
    logger.info(f"context_vecs.shape: {context_vecs.shape}")  # [batch_size, context_len, embed_dim // 12 * n_heads]
    # ------------------------------
    # The multi-head attention
    # ------------------------------
    mha = MultiHeadAttention(
        d_model=embed_dim, 
        d_out=embed_dim, 
        n_heads=n_heads,
        context_length=context_len, 
        dropout=0.0, 
        qkv_bias=False
    ).to(device)
    context_vecs = mha(embeddings)
    # logger.info(f"context_vecs: \n{context_vecs}")
    logger.info(f"context_vecs.shape: {context_vecs.shape}")
    # ------------------------------
    # An alternative multi-head attention with combined weights
    # ------------------------------  
    mha_combined_qkv = MultiHeadAttentionCombinedQKV(
        d_model=embed_dim,
        d_out=embed_dim,
        n_heads=n_heads,
        context_length=context_len,
        dropout=0.0,
        qkv_bias=False
    ).to(device)
    context_vecs = mha_combined_qkv(embeddings)
    # logger.info(f"context_vecs: \n{context_vecs}")
    logger.info(f"context_vecs.shape: {context_vecs.shape}")
    # ------------------------------
    # Multi-head attention with Einsum
    # ------------------------------
    mha_einsum = MHAEinsum(
        d_model=embed_dim,
        d_out=embed_dim,
        n_heads=n_heads,
        context_length=context_len,
        dropout=0.0,
        qkv_bias=False
    ).to(device)
    context_vecs = mha_einsum(embeddings)
    # logger.info(f"context_vecs: \n{context_vecs}")
    logger.info(f"context_vecs.shape: {context_vecs.shape}")
    # ------------------------------
    # Multi-head attention with PyTorch's scaled dot product attention and FlashAttention
    # ------------------------------
    mha_pytorch_scaled = MHAPyTorchScaledDotProduct(
        d_model=embed_dim,
        d_out=embed_dim,
        n_heads=n_heads,
        context_length=context_len,
        dropout=0.0,
        qkv_bias=False
    ).to(device)
    context_vecs = mha_pytorch_scaled(embeddings)
    # logger.info(f"context_vecs: \n{context_vecs}")
    logger.info(f"context_vecs.shape: {context_vecs.shape}")
    # ------------------------------
    # PyTorch's scaled dot product attention without FlashAttention
    # ------------------------------
    mha_pytorch_sdpa_no_flash = MHAPyTorchSDPAWithoutFlash(
        d_model=embed_dim,
        d_out=embed_dim,
        n_heads=n_heads,
        context_length=context_len,
        dropout=0.0,
        qkv_bias=False
    ).to(device)
    context_vecs = mha_pytorch_sdpa_no_flash(embeddings)
    # logger.info(f"context_vecs: \n{context_vecs}")
    logger.info(f"context_vecs.shape: {context_vecs.shape}")
    # ------------------------------
    # Using PyTorch's torch.nn.MultiheadAttention
    # ------------------------------
    mha_pytorch_class_default = MHAPyTorchClass(
        d_model=embed_dim,
        d_out=embed_dim,
        n_heads=n_heads,
        context_length=context_len,
        dropout=0.0,
        qkv_bias=False
    ).to(device)
    context_vecs = mha_pytorch_class_default(embeddings)
    # logger.info(f"context_vecs: \n{context_vecs}")
    logger.info(f"context_vecs.shape: {context_vecs.shape}")
    # ------------------------------
    # Using PyTorch's torch.nn.MultiheadAttention with scaled_dot_product_attention
    # ------------------------------
    mha_pytorch_class_noweights = MHAPyTorchClass(
        d_model=embed_dim,
        d_out=embed_dim,
        n_heads=n_heads,
        context_length=context_len,
        dropout=0.0,
        qkv_bias=False,
        need_weights=False # NEW!
    ).to(device)
    context_vecs = mha_pytorch_class_noweights(embeddings)
    # logger.info(f"context_vecs: \n{context_vecs}")
    logger.info(f"context_vecs.shape: {context_vecs.shape}")
    # ------------------------------
    # Using PyTorch's FlexAttention
    # ------------------------------
    mha_pytorch_flex = MHAPyTorchFlexAttention(
        d_model=embed_dim,
        d_out=embed_dim,
        n_heads=n_heads,
        context_length=context_len,
        dropout=0.0,
        qkv_bias=False
    ).to(device)
    context_vecs = mha_pytorch_flex(embeddings)
    # logger.info(f"context_vecs: \n{context_vecs}")
    logger.info(f"context_vecs.shape: {context_vecs.shape}")
    # ------------------------------
    # method 3: MultiHeadAttentionRoPE
    # ------------------------------
    mha_rope = MultiHeadAttentionRoPE(
        d_model=embed_dim,
        d_out=embed_dim,
        n_heads=n_heads,
        context_length=context_len,
    ).to(device)
    mha_rope(embeddings)
    logger.info(f"W_key: {mha_rope.W_key.weight.shape}")
    logger.info(f"W_value: {mha_rope.W_value.weight.shape}")
    logger.info(f"W_query: {mha_rope.W_query.weight.shape}")

    # ------------------------------
    # method 4: GroupedQueryAttention
    # ------------------------------
    gqa = GroupedQueryAttention(
        d_model=embed_dim,
        d_out=embed_dim,
        n_heads=n_heads,
        context_length=context_len,
        num_kv_groups=4,
        rope_base=llama3_theta_base,
    ).to(device)
    gqa(embeddings)
    logger.info(f"W_key: {gqa.W_key.weight.shape}")
    logger.info(f"W_value: {gqa.W_value.weight.shape}")
    logger.info(f"W_query: {gqa.W_query.weight.shape}")
    # ------------------------------
    # number of params
    # ------------------------------
    logger.info(f"Total number of parameters:")
    mha_total_params = sum(p.numel() for p in mha.parameters())
    logger.info(f"MHA: {mha_total_params}")

    gqa_total_params = sum(p.numel() for p in gqa.parameters())
    logger.info(f"GQA: {gqa_total_params}")
    
    # free up memory
    del mha
    del gqa 

if __name__ == "__main__":
    main()
