# -*- coding: utf-8 -*-

# ***************************************************
# * File        : gpt_pretrain_test.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-02-14
# * Version     : 0.1.021423
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
import torch.nn as nn

from data_provider.pretrain.data_load import data_load
from tokenizers.tokenization import text_to_token_ids, token_ids_to_text
from models.gpt import Model
from utils.llm.gpt_generate import generate_text_simple
from utils.device import device_setting
from utils.args_tools import DotDict
from utils.log_util import logger

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]

# device
device = device_setting()


# 测试代码 main 函数
def main():
    # ------------------------------
    # model
    # ------------------------------
    # params
    GPT_CONFIG_124M = {
        "vocab_size": 50257,   # Vocabulary size
        "context_length": 256, # Shortened context length (orig: 1024)
        "emb_dim": 768,        # Embedding dimension
        "n_heads": 12,         # Number of attention heads
        "n_layers": 12,        # Number of layers
        "dropout": 0.1,      # Dropout rate
        "qkv_bias": False      # Query-key-value bias
    }
    GPT_CONFIG_124M = DotDict(GPT_CONFIG_124M)
    train_cfgs = {
        "train_epochs": 10,
        "max_new_tokens": 10,
    }

    # seed
    torch.manual_seed(123)

    # model 
    model = Model(GPT_CONFIG_124M)

    # disable dropout durning inference
    model.eval()
    
    # ------------------------------
    # text generation
    # ------------------------------
    # text
    start_context = "Every effort move you"
    
    # token IDs 
    token_ids = generate_text_simple(
        model = model,
        token_idx = text_to_token_ids(start_context),
        max_new_tokens = 10,
        context_size = GPT_CONFIG_124M.context_length,
    )
    logger.info(f"Output text: \n{token_ids_to_text(token_ids)}")
    
    # ------------------------------
    # text generation loss: cross-entropy and perplexity
    # ------------------------------
    # input data
    inputs = torch.tensor([[16833, 3626, 6100],   # ["every effort moves",
                           [40,    1107, 588]])   #  "I really like"]
    logger.info(f"inputs shape: {inputs.shape}")
    
    # target data
    targets = torch.tensor([[3626, 6100, 345  ],  # [" effort moves you",
                            [1107,  588, 11311]]) #  " really like chocolate"]
    logger.info(f"targets shape: {targets.shape}")
    
    # model inference
    with torch.no_grad():
        logits = model(inputs)
    logger.info(f"logits: \n{logits} \nlogits shape: {logits.shape}")
    
    # softmax
    probas = torch.softmax(logits, dim=-1)
    logger.info(f"probas: \n{probas} \nprobas.shape: {probas.shape}")
    
    # argmax
    token_ids = torch.argmax(probas, dim=-1, keepdim=True)
    logger.info(f"Token IDs: \n{token_ids} \nToken IDs shape: {token_ids.shape}")
    logger.info(f"Targets batch 1: \n{targets[0]} \n{token_ids_to_text(targets[0])}")
    logger.info(f"Outputs batch 1: \n{token_ids[0].flatten()} \n{token_ids_to_text(token_ids[0].flatten())}")
    logger.info(f"Targets batch 1: \n{targets[1]} \n{token_ids_to_text(targets[1])}")
    logger.info(f"Outputs batch 1: \n{token_ids[1].flatten()} \n{token_ids_to_text(token_ids[1].flatten())}")

    # token probabilities corresponding to the target indices
    text_idx = 0
    target_probas_1 = probas[text_idx, [0, 1, 2], targets[text_idx]]
    logger.info(f"Text 1: \ntargets[text_idx]: {targets[text_idx]} \ntarget_probas_1: {target_probas_1}")
    text_idx = 1
    target_probas_2 = probas[text_idx, [0, 1, 2], targets[text_idx]]
    logger.info(f"Text 2: \ntargets[text_idx]: {targets[text_idx]} \ntarget_probas_2: {target_probas_2}")
    
    # compute logarithm of all token probabilities
    log_probas = torch.log(torch.cat([target_probas_1, target_probas_2]))
    logger.info(f"log_probas: \n{log_probas}")
    
    avg_log_probas = torch.mean(log_probas)
    logger.info(f"avg log probas: {avg_log_probas}")
    
    neg_avg_log_probas = avg_log_probas * -1
    logger.info(f"neg_avg_log_probas: {neg_avg_log_probas}")
    
    # Logits have shape: (batch_size, num_tokens, vocab_size)
    logger.info(f"Logits shape: {logits.shape}")
    # Targets have shape: (batch_size, num_tokens)
    logger.info(f"Targets shape: {targets.shape}")
    
    # compute cross-entropy loss
    logits_flat = logits.flatten(0, 1)
    logger.info(f"logits_flat: \n{logits_flat} \nFlattened logtis: {logits_flat.shape}")
    targets_flat = targets.flatten(0, 1)
    logger.info(f"targets_flat: \n{targets_flat} \nFlattened targets: {targets_flat.shape}")
    
    loss = nn.CrossEntropyLoss()
    # loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)
    loss = loss(logits_flat, targets_flat)
    logger.info(f"loss: {loss}")
    # perplexity
    perplexity = torch.exp(loss)
    logger.info(f"perplexity: {perplexity}")

    # ------------------------------
    # model training
    # ------------------------------
    # 数据加载
    raw_text = data_load(
        url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"
    )
    logger.info(f"raw_text[:99]: \n{raw_text[:99]}")
    logger.info(f"raw_text[:99]: \n{raw_text[-99:]}")

    # Sanity check
    total_characters = len(raw_text)
    import tiktoken
    tokenizer = tiktoken.get_encoding("gpt2")
    total_tokens = len(tokenizer.encode(raw_text))
    logger.info(f"total_characters: {total_characters}")
    logger.info(f"total_tokens: {total_tokens}")
 
    # training and validation data loader create
    from data_provider.pretrain.data_loader import create_dataloader
    train_ratio = 0.90
    split_idx = int(train_ratio * len(raw_text))
    train_data = raw_text[:split_idx]
    val_data = raw_text[split_idx:]
    if total_tokens * (train_ratio) < GPT_CONFIG_124M.context_length:
        print("Not enough tokens for the training loader. "
            "Try to lower the `GPT_CONFIG_124M['context_length']` or "
            "increase the `training_ratio`")

    if total_tokens * (1-train_ratio) < GPT_CONFIG_124M.context_length:
        print("Not enough tokens for the validation loader. "
            "Try to lower the `GPT_CONFIG_124M['context_length']` or "
            "decrease the `training_ratio`")
    
    train_loader = create_dataloader(
        train_data,
        batch_size=2,
        max_length=GPT_CONFIG_124M.context_length,
        stride=GPT_CONFIG_124M.context_length,
        drop_last=True,
        shuffle=True,
        num_workers=0,
    )
    val_loader = create_dataloader(
        val_data,
        batch_size=2,
        max_length=GPT_CONFIG_124M.context_length,
        stride=GPT_CONFIG_124M.context_length,
        drop_last=False,
        shuffle=False,
        num_workers=0,
    ) 
    logger.info(f"Train loader:")
    for x, y in train_loader:
        logger.info(f"x.shape: {x.shape}, y.shape: {y.shape}")
    logger.info("Validation loader:")
    for x, y in val_loader:
        logger.info(f"x.shape: {x.shape}, y.shape: {y.shape}")
    
    # train, validation tokens
    train_tokens = 0
    for input_batch, target_batch in train_loader:
        train_tokens += input_batch.numel()
    val_tokens = 0
    for input_batch, target_batch in val_loader:
        val_tokens += input_batch.numel()
    logger.info(f"Training tokens: {train_tokens}")
    logger.info(f"Validation tokens: {val_tokens}")
    logger.info(f"All tokens: {train_tokens + val_tokens}")
 
    # model
    def _calc_loss_batch(input_batch, target_batch, model, device):
        # criterion
        criterion = nn.CrossEntropyLoss()
        # training data batch
        input_batch = input_batch.to(device)
        target_batch = target_batch.to(device)
        # forward
        logits = model(input_batch)
        # loss
        loss = criterion(logits.flatten(0, 1), target_batch.flatten(0, 1))

        return loss

    def _calc_loss_loader(data_loader, model, device, num_batches = None):
        # number of batches
        if len(data_loader) == 0:
            return float("nan")
        elif num_batches is None:
            num_batches = len(data_loader)
        else:
            num_batches = min(num_batches, len(data_loader))
        # calculate loss
        total_loss = 0
        for i, (input_batch, target_batch) in enumerate(data_loader):
            if i < num_batches:
                loss = _calc_loss_batch(input_batch, target_batch, model, device)
                total_loss += loss.item()
            else:
                break
        
        return total_loss / num_batches
    # seed
    torch.manual_seed(123)
    # model
    model = Model(GPT_CONFIG_124M).to(device)
    # loss
    with torch.no_grad():
        train_loss = _calc_loss_loader(train_loader, model, device)
        val_loss = _calc_loss_loader(val_loader, model, device)
    logger.info(f"Training loss: {train_loss}")
    logger.info(f"Validation loss: {val_loss}") 

if __name__ == "__main__":
    main()
