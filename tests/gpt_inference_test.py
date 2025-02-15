# -*- coding: utf-8 -*-

# ***************************************************
# * File        : inference.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-01-29
# * Version     : 1.0.012900
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

import tiktoken
import torch
import matplotlib.pyplot as plt

from utils.log_util import logger

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]



# 测试代码 main 函数
def main():
    from models.gpt import Model
    from layers.tokenization import text_to_token_ids, token_ids_to_text
    from models.gpt_generate import generate_text_simple, generate
    from utils.argsparser_tools import DotDict
    from utils.device import device
    # ------------------------------
    # model params
    # ------------------------------
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
    GPT_CONFIG_124M = DotDict(GPT_CONFIG_124M)

    # ------------------------------
    # model
    # ------------------------------
    # model
    torch.manual_seed(123)
    model = Model(GPT_CONFIG_124M).to(device)
    
    # ------------------------------
    # model simple inference
    # ------------------------------
    logger.info(f"{50 * '='}")
    logger.info(f"Model simple inference")
    logger.info(f"{50 * '='}")
    # generate_text simple
    model.to("cpu")
    model.eval()
    token_ids = generate_text_simple(
        model = model,
        token_idx = text_to_token_ids("Every effort moves you"),
        max_new_tokens = 25,
        context_size = GPT_CONFIG_124M.context_length,
    )
    logger.info(f"Output text: \n{token_ids_to_text(token_ids)}")
    # ------------------------------
    # temperature scaling and top-k decoding strategies
    # ------------------------------
    # vocab and inverse vocab
    vocab = {
        "closer": 0,
        "every": 1,
        "effort": 2,
        "forward": 3,
        "inches": 4,
        "moves": 5,
        "pizza": 6,
        "toward": 7,
        "you": 8,
    }
    inverse_vocab = {v: k for k, v in vocab.items()}

    # input: "every effort moves you", LLM returns the following logits for the next token
    next_token_logits = torch.tensor([4.51, 0.89, -1.90, 6.75, 1.63, -1.62, -1.89, 6.28, 1.79])
    logger.info(f"{50 * '='}")
    logger.info("input: 'every effort moves you', LLM returns the following logits for the next token")
    logger.info(f"{50 * '='}")
    logger.info(f"Next token logits: \n{next_token_logits}")
    # ------------------------------
    # decoding strategies: temperature scaling(add variety)
    # ------------------------------
    logger.info(f"{50 * '='}")
    logger.info(f"Decoding strategies: temperature scaling(add variety)")
    logger.info(f"{50 * '='}") 

    # method 1: torch.argmax
    # ----------------------------
    logger.info(f"{25 * '-'}")
    logger.info(f"# method 1: torch.argmax")
    logger.info(f"{25 * '-'}")
    probas = torch.softmax(next_token_logits, dim = 0)
    logger.info(f"Next token's probas: \n{probas}")
    next_token_id = torch.argmax(probas).item()
    logger.info(f"next generated token id: {next_token_id}")
    logger.info(f"next generated token: {inverse_vocab[next_token_id]}")

    # method 2: torch.multinomial
    # ----------------------------
    logger.info(f"{25 * '-'}")
    logger.info(f"# method 2: torch.multinomial")
    logger.info(f"{25 * '-'}")
    def print_sampled_token(probas):
        torch.manual_seed(123)
        # sample the next token 1,000 times using the original softmax probabilities
        sample = [
            torch.multinomial(probas, num_samples = 1).item()
            for i in range(1_000)
        ]
        sampled_ids = torch.bincount(torch.tensor(sample))
        logger.info(f"next generated token:")
        for i, freq in enumerate(sampled_ids):
            logger.info(f"{freq} x {inverse_vocab[i]}")
    
    probas = torch.softmax(next_token_logits, dim = 0)
    logger.info(f"Next token's probas: \n{probas}")
    print_sampled_token(probas)

    # method 3: softmax with temperature
    # ----------------------------
    logger.info(f"{25 * '-'}")
    logger.info(f"# method 3: softmax with temperature")
    logger.info(f"{25 * '-'}")
    def softmax_with_temperature(logits, temperature):
        scaled_logits = logits / temperature

        return torch.softmax(scaled_logits, dim = 0)

    # temperature values
    # 1.0: origiinal
    # 0.1: higher confidence
    # 5.0: lower confidence
    temperatures = [1, 0.1, 5]
    scaled_probas = [
        softmax_with_temperature(next_token_logits, T)
        for T in temperatures
    ]
    logger.info(f"scaled_probas: \n{scaled_probas}")

    # Plotting
    """
    x = torch.arange(len(vocab))
    bar_width = 0.15
    fig, ax = plt.subplots(figsize=(5, 3))
    for i, T in enumerate(temperatures):
        rects = ax.bar(
            x + i * bar_width, 
            scaled_probas[i], 
            bar_width, 
            label=f'Temperature = {T}'
        )
    ax.set_ylabel('Probability')
    ax.set_xticks(x)
    ax.set_xticklabels(vocab.keys(), rotation=90)
    ax.legend()
    plt.tight_layout()
    # plt.savefig("temperature-plot.pdf")
    plt.show()
    """

    # rescaled probabilities via temperature 0.1
    print_sampled_token(scaled_probas[1])

    # rescaled probabilities via temperature 5
    print_sampled_token(scaled_probas[2])

    # ------------------------------
    # decoding strategies: top-k sampling
    # ------------------------------
    logger.info(f"{50 * '='}")
    logger.info(f"Decoding strategies: top-k sampling")
    logger.info(f"{50 * '='}")
    top_k = 3
    top_logits, top_pos = torch.topk(next_token_logits, top_k)
    logger.info(f"Top logits: \n{top_logits}")
    logger.info(f"Top positions: \n{top_pos}")

    # create tensor containing -inf values
    new_logits = torch.where(
        condition=next_token_logits < top_logits[-1],
        input=torch.tensor(float("-inf")),
        other=next_token_logits,
    )
    # or
    new_logits = torch.full_like(next_token_logits, -torch.inf)

    # copy top k values into the -inf tensor
    new_logits[top_pos] = next_token_logits[top_pos]
    logger.info(f"new_logits: \n{new_logits}")

    topk_probas = torch.softmax(new_logits, dim=0)
    logger.info(f"topk_probas: \n{topk_probas}")
    
    # ------------------------------
    # Model inference
    # ------------------------------
    logger.info(f"{50 * '='}")
    logger.info(f"Model inference")
    logger.info(f"{50 * '='}")
    # generate_text simple
    model.to("cpu")
    model.eval()
    token_ids = generate(
        model = model,
        token_idx = text_to_token_ids("Every effort moves you"),
        max_new_tokens = 25,
        context_size = GPT_CONFIG_124M.context_length,
    )
    logger.info(f"Output text: \n{token_ids_to_text(token_ids)}")

    
    """
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
    GPT_CONFIG_124M = DotDict(GPT_CONFIG_124M)
    
    # tokenizer
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
    
    # ------------------------------
    # GPT model
    # ------------------------------
    torch.manual_seed(123)
    # model
    model = Model(GPT_CONFIG_124M)
    model.to(device)
    
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
        token_idx = encoded_tensor,
        max_new_tokens = 6,
        context_size=GPT_CONFIG_124M.context_length,
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
        token_idx = text_to_token_ids("Every effort moves you"),
        max_new_toknes = 15,
        context_size = GPT_CONFIG_124M.context_length,
        top_k = 25,
        temperature = 1.4,
    )
    logger.info(f"Output text: \n{token_ids_to_text(token_ids)}")
    """

if __name__ == "__main__":
    main()
