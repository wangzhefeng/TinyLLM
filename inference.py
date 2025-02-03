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

# from tiny_model.TinyLLM.training import (
#     text_to_token_ids, 
#     token_ids_to_text,
#     generate_text_simple,
#     GPT_CONFIG_124M,
# )
from utils.log_util import logger

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


# ------------------------------
# inference
# ------------------------------
# model = None
# model.to("cpu")
# model.eval()
# tokenizer = tiktoken.get_encoding("gpt2")
# token_ids = generate_text_simple(
#     model = model,
#     idx = text_to_token_ids("Every effort moves you", tokenizer),
#     max_new_tokens = 25,
#     context_size = GPT_CONFIG_124M["context_length"],
# )
# logger.info(f"Output text: \n{token_ids_to_text(token_ids, tokenizer)}")




# 测试代码 main 函数
def main():
    # ------------------------------
    # decoding strategies: temperature scaling(add variety)
    # ------------------------------
    # vocab
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
    probas = torch.softmax(next_token_logits, dim = 0)
    logger.info(f"probas: \n{probas}")

    # method 1: torch.argmax
    # ----------------------------
    logger.info(f"# method 1: torch.argmax")
    logger.info("-" * 40)
    next_token_id = torch.argmax(probas).item()
    logger.info(f"next generated token: {inverse_vocab[next_token_id]}")

    # method 2: torch.multinomial
    # ----------------------------
    logger.info(f"# method 2: torch.multinomial")
    logger.info("-" * 40)
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

    print_sampled_token(probas)

    # method 3: softmax with temperature
    # ----------------------------
    logger.info(f"# method 3: softmax with temperature")
    logger.info("-" * 40)
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
    # plt.show()

    # rescaled probabilities via temperature 0.1
    print_sampled_token(scaled_probas[1])

    # rescaled probabilities via temperature 5
    print_sampled_token(scaled_probas[2])


    # ------------------------------
    # decoding strategies: top-k sampling
    # ------------------------------
    top_k = 3
    top_logits, top_pos = torch.topk(next_token_logits, top_k)
    logger.info(f"Top logits: \n{top_logits}")
    logger.info(f"Top positions: \n{top_pos}")

    new_logits = torch.where(
        condition=next_token_logits < top_logits[-1],
        input=torch.tensor(float("-inf")),
        other=next_token_logits,
    )
    # create tensor containing -inf values
    new_logits = torch.full_like(next_token_logits, -torch.inf)
    # copy top k values into the -inf tensor
    new_logits[top_pos] = next_token_logits[top_pos]
    logger.info(f"new_logits: \n{new_logits}")

    topk_probas = torch.softmax(new_logits, dim=0)
    logger.info(f"topk_probas: \n{topk_probas}")

if __name__ == "__main__":
    main()
