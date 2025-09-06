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
# ***************************************************

__all__ = []

# python libraries
import os
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)
import time

import torch

from models.gpt2_124M import Model
from layers.tokenizers.tokenization import (
    text_to_token_ids, 
    token_ids_to_text,
)
from layers.generator import generate_simple, generate_simple_cached, generate

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]
os.environ['LOG_NAME'] = LOGGING_LABEL
from utils.log_util import logger


def gpt2_124M_model_inference_test(tokenizer, GPT2_124M_CONFIG, device, temperature: float=None, top_k: float=None, eos_id: int=None):
    # input data
    start_context = "Hello, I am"
    logger.info(f"Input text: {start_context}")

    # input tokenization
    token_ids = tokenizer.encode(start_context)
    token_ids_tensor = torch.tensor(token_ids).unsqueeze(0).to(device)
    logger.info(f"Encoded input text: {token_ids_tensor}")
    logger.info(f"Encoded input shape: {token_ids_tensor.shape}")

    # model
    model = Model(GPT2_124M_CONFIG).to(device)
    # disable dropout
    model.eval()
    # ------------------------------
    # inference
    # ------------------------------
    # inference start
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_time =time.time()

    # generate text
    if temperature and top_k and eos_id:
        out = generate(
            model = model,
            token_idx = token_ids_tensor,
            max_new_tokens = GPT2_124M_CONFIG.max_new_toknes,
            context_length = GPT2_124M_CONFIG.context_length,
            temperature = temperature,
            top_k = top_k,
            eos_id = eos_id,
            use_cache=True,
        )
    else:
        # out = generate_simple(
        #     model = model,
        #     token_idx = token_ids_tensor,
        #     max_new_tokens = GPT2_124M_CONFIG.max_new_toknes,
        #     context_length = GPT2_124M_CONFIG.context_length,
        # )
        out = generate_simple_cached(
            model = model,
            token_idx = token_ids_tensor,
            max_new_tokens = GPT2_124M_CONFIG.max_new_toknes,
            context_length = GPT2_124M_CONFIG.context_length,
            use_cache=True,
        )
    logger.info(f"Output: {out}")
    logger.info(f"Outout shape: {out.shape}")
    # inference end
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    total_time = time.time() - start_time

    # remove batch dimension and convert back into text
    decoded_text = tokenizer.decode(out.squeeze(0).tolist())
    logger.info(f"Output text: {decoded_text}")

    # inference performance
    logger.info(f"Time: {total_time:.2f} sec")
    logger.info(f"{int(len(out[0]) / total_time)} tokens/sec")
    if torch.cuda.is_available():
        max_memory_bytes = torch.cuda.max_memory_allocated()
        max_memory_gb = max_memory_bytes / (1024 ** 3)
        logger.info(f"Max memory allocated: {max_memory_gb:.2f} GB")


def test_todo(model, tokenizer, GPT2_124M_CONFIG):
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
        context_length = GPT2_124M_CONFIG.context_length,
    )
    logger.info(f"Output text: \n{token_ids_to_text(token_ids)}")

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
    
    # model forward
    logits = model(batch)
    logger.info(f"Input: \n{batch}")
    logger.info(f"Output: \n{logits}")
    logger.info(f"Output shape: {logits.shape}")
    
    # generating text: v2
    token_ids = generate(
        model = model,
        token_idx = text_to_token_ids("Every effort moves you"),
        max_new_toknes = 15,
        context_length = GPT2_124M_CONFIG.context_length,
        top_k = 25,
        temperature = 1.4,
    )
    logger.info(f"Output text: \n{token_ids_to_text(token_ids)}")



 
# 测试代码 main 函数
def main():
    from test.gpt_model.model_config import device, tokenizer, GPT2_124M_CONFIG
    
    gpt2_124M_model_inference_test(tokenizer, GPT2_124M_CONFIG, device)

if __name__ == "__main__":
    main()
