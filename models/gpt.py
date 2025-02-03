# -*- coding: utf-8 -*-

# ***************************************************
# * File        : gpt.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-01-25
# * Version     : 1.0.012519
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

import torch
import torch.nn as nn

from tiny_model.TinyLLM.utils.activation import GELU
from tiny_model.TinyLLM.attention import MultiHeadAttention
from utils.log_util import logger

# set options
# torch.set_printoptions(sci_mode=False)

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


class GPTModel(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        # Embedding
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])  #TODO
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        # TransformerBlock
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        # LayerNorm
        self.final_norm = LayerNorm(cfg["emb_dim"])
        # output head Linear
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias = False)

    def forward(self, in_idx):
        # in_idx size
        batch_size, seq_len = in_idx.shape
        # embedding
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds  # shape: [batch_size, num_tokens, emb_size]
        x = self.drop_emb(x)  # dropout
        # transformer blocks
        x = self.trf_blocks(x)
        # final norm
        x = self.final_norm(x)
        # output head
        logits = self.out_head(x)

        return logits


class TransformerBlock(nn.Module):
    
    def __init__(self, cfg):
        super().__init__()
        
        self.attn = MultiHeadAttention(
            d_in = cfg["emb_dim"],
            d_out = cfg["emb_dim"],
            context_length = cfg["context_length"],
            num_heads = cfg["n_heads"],
            dropout = cfg["drop_rate"],
            qkv_bias = cfg["qkv_bias"],
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])
    
    def forward(self, x):
        # shortcut connection for attention block
        shortcut = x
        x = self.norm1(x)
        x = self.attn(x)
        x = self.drop_shortcut(x)
        x = shortcut + x
        # shortcut connection for feed forward block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = shortcut + x
        
        return x


class LayerNorm(nn.Module):
    
    def __init__(self, emb_dim, eps = 1e-5):
        super().__init__()

        self.eps = eps
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class FeedForward(nn.Module):
    
    def __init__(self, cfg):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )
    
    def forward(self, x):
        return self.layers(x)


def generate_text_simple(model, idx: torch.tensor, max_new_tokens: int, context_size: int):
    # idx is (batch, n_tokens) array of indices in the current contex
    for _ in range(max_new_tokens):
        # crop current context if it exceeds the supported context size
        idx_cond = idx[:, -context_size:]
        # get the predictions
        with torch.no_grad():
            logits = model(idx_cond)
        # focus only on the last time step
        # (batch, n_tokens, vocab_size) -> (batch, vocab_size)
        logits = logits[:, -1, :]
        logger.info(f"logits: {logits}")
        # softmax
        probas = torch.softmax(logits, dim=-1)  # (batch, vocab_size)
        logger.info(f"probas: {probas}")
        # get the idx of the vocab entry with the highest probability value
        idx_next = torch.argmax(probas, dim = -1, keepdim = True)
        logger.info(f"idx_next: {idx_next}")
        # append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim = 1)  # (batch, n_tokens+1)
        logger.info(f"idx: {idx}\n")

    return idx


def generate(model, idx, max_new_tokens, context_size, 
             temperature=0.0, top_k=None, eos_id=None):
    # for-loop is the same as before: get logits, and only focus on last time step
    for _ in range(max_new_tokens):
        # crop current context if it exceeds the supported context size
        idx_cond = idx[:, -context_size:]
        # get the predictions
        with torch.no_grad():
            logits = model(idx_cond)
        # focus only on the last time step
        logits = logits[:, -1, :]  # (batch, n_tokens, vocab_size) -> (batch, vocab_size)
        # filter logits with top_k sampling
        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(
                logits < min_val, 
                torch.tensor(float("-inf")).to(logits.device),
                logits
            )
        # apply temperature scaling
        if temperature > 0.0:
            logits = logits / temperature
            # apply softmax to get probabilities
            probs = torch.softmax(logits, dim=-1)  # (batch_size, context_len)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)
        # otherwise same as before: get idx of the vocab entry with the highest logits value
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch_size, 1)
        # stop generating early if end-of-sequence token is encountered and eos_id is specified
        if idx_next == eos_id:
            break
        # append sampled index to the running sequence
        idx = torch.cat([idx, idx_next], dim=1)  # (batch_size, num_tokens+1)

    return idx






# 测试代码 main 函数
def main():
    GPT_CONFIG_124M = {
        "vocab_size": 50257,
        "context_length": 1024,
        "emb_dim": 768,
        "n_heads": 12,
        "n_layers": 12,
        "drop_rate": 0.1,
        "qkv_bias": False,
    }
    
    # tokenizer
    import tiktoken
    tokenizer = tiktoken.get_encoding("gpt2")
    
    # input data
    batch = []
    text1 = "Every effort moves you"
    text2 = "Every day hold a"
    batch.append(torch.tensor(tokenizer.encode(text1)))
    batch.append(torch.tensor(tokenizer.encode(text2)))
    batch = torch.stack(batch, dim=0)
    logger.info(f"batch: \n{batch}")
    logger.info(f"batch.shape: {batch.shape}")

    # ------------------------------
    # Layer Norm test
    # ------------------------------
    # data
    torch.manual_seed(123)
    batch_example = torch.randn(2, 5)
    logger.info(f"batch_example: \n{batch_example}")

    ln = LayerNorm(emb_dim=5)
    out_ln = ln(batch_example)
    logger.info(f"out_ln: \n{out_ln}")
    mean = out_ln.mean(dim=-1, keepdim=True)
    var = out_ln.var(dim=-1, unbiased=False, keepdim=True)
    logger.info(f"Mean: \n{mean}")
    logger.info(f"Variance: \n{var}")
    
    # ------------------------------
    # Feed Forward test
    # ------------------------------
    ffn = FeedForward(GPT_CONFIG_124M)
    x = torch.rand(2, 3, 768)
    out = ffn(x)
    logger.info(out.shape)

    # ------------------------------
    # Transformer Block test
    # ------------------------------
    # shape: [batch_size, num_tokens, emb_dim]
    torch.manual_seed(123)
    x = torch.rand(2, 4, 768)
    block = TransformerBlock(GPT_CONFIG_124M)
    output = block(x)
    logger.info(f"Input shape: {x.shape}")
    logger.info(f"Output shape: {output.shape}")

    # ------------------------------
    # GPT model
    # ------------------------------
    torch.manual_seed(123)
    # model
    model = GPTModel(GPT_CONFIG_124M)
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
    # token_ids = generate(
    #     model = model,
    #     idx = text_to_token_ids("Every effort moves you", tokenizer),
    #     max_new_toknes = 15,
    #     context_size = GPT_CONFIG_124M["context_length"],
    #     top_k = 25,
    #     temperature = 1.4,
    # )
    # logger.info(f"Output text: \n{token_ids_to_text(token_ids, tokenizer)}")

if __name__ == "__main__":
    main()
