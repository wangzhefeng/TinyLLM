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

from data_provider.data_loader import load_local_data
from data_provider.data_loader import create_dataloader
from layers.tokenizers.tokenization import (
    choose_tokenizer,
    text_to_token_ids,
    token_ids_to_text,
)
from models.gpt2.gpt2_124M import Model
from utils.llm.calc_loss import calc_loss_loader

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]
os.environ['LOG_NAME'] = LOGGING_LABEL
from utils.log_util import logger


def text_generation(GPT2_CONFIG, device):
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
    # ------------------------------
    # model
    # ------------------------------
    model = Model(GPT2_CONFIG).to(device)
    # ------------------------------
    # model inference
    # ------------------------------
    with torch.no_grad():
        logits = model(inputs.to(device))
    logger.info(f"logits: \n{logits} \nlogits shape: {logits.shape}")
    
    # softmax
    probas = torch.softmax(logits, dim=-1)
    logger.info(f"probas: \n{probas} \nprobas.shape: {probas.shape}")

    # argmax
    output_token_ids = torch.argmax(probas, dim=-1, keepdim=True)
    logger.info(f"\nOutput Token IDs: \n{output_token_ids} \nOutput Token IDs shape: {output_token_ids.shape}")
    logger.info(f"\nTargets batch 0: {targets[0]}             \nTarget batch 0 text: {token_ids_to_text(targets[0])}")
    logger.info(f"\nOutputs batch 0: {output_token_ids[0].flatten()} \nOutputs batch 0 text: {token_ids_to_text(output_token_ids[0].flatten())}")
    logger.info(f"\nTargets batch 1: {targets[1]}             \nTarget batch 1 text: {token_ids_to_text(targets[1])}")
    logger.info(f"\nOutputs batch 1: {output_token_ids[1].flatten()} \nOutputs batch 1 text: {token_ids_to_text(output_token_ids[1].flatten())}")

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
    
    # ------------------------------
    # loss
    # ------------------------------
    # compute cross-entropy loss
    logits_flat = logits.flatten(0, 1).to(device)
    logger.info(f"logits_flat: \n{logits_flat} \nFlattened logtis: {logits_flat.shape}")
    targets_flat = targets.flatten(0, 1).to(device)
    logger.info(f"targets_flat: \n{targets_flat} \nFlattened targets: {targets_flat.shape}")
    
    # loss
    criterion = nn.CrossEntropyLoss()
    loss = criterion(logits_flat, targets_flat)
    logger.info(f"loss: {loss}")
    
    # perplexity
    perplexity = torch.exp(loss)
    logger.info(f"perplexity: {perplexity}")


def test_training(tokenizer, GPT2_CONFIG, device):
    # ------------------------------
    # model training
    # ------------------------------
    # data path
    data_path = "./dataset/pretrain/gpt"
    data_file = "the-verdict.txt" 
    # 数据加载
    raw_text = load_local_data(data_path=data_path, data_file=data_file)
    logger.info(f"raw_text[:99]: \n{raw_text[:99]}")
    logger.info(f"raw_text[:99]: \n{raw_text[-99:]}")

    # Sanity check
    total_characters = len(raw_text)
    total_tokens = len(tokenizer.encode(raw_text))
    logger.info(f"total_characters: {total_characters}")
    logger.info(f"total_tokens: {total_tokens}")

    # training and validation data loader create
    train_ratio = 0.90
    split_idx = int(train_ratio * len(raw_text))
    train_data = raw_text[:split_idx]
    val_data = raw_text[split_idx:]
    if total_tokens * (train_ratio) < GPT2_CONFIG.context_length:
        logger.info("Not enough tokens for the training loader. "
                    "Try to lower the `GPT2_124M_CONFIG['context_length']` or "
                    "increase the `training_ratio`")

    if total_tokens * (1-train_ratio) < GPT2_CONFIG.context_length:
        logger.info("Not enough tokens for the validation loader. "
                    "Try to lower the `GPT2_124M_CONFIG['context_length']` or "
                    "decrease the `training_ratio`")
    
    # dataloader
    torch.manual_seed(123)
    train_dataset, train_dataloader = create_dataloader(
        data_source="local",
        data_path = data_path, 
        data_file = data_file,
        flag = "train", 
        train_ratio = 0.8,
        tokenizer = tokenizer,
        batch_size = 2,
        max_len = 256,
        num_workers = 0,
        device=device,
    )
    valid_dataset, valid_dataloader = create_dataloader(
        data_source="local",
        data_path = data_path, 
        data_file = data_file,
        flag = "valid", 
        train_ratio = 0.8,
        tokenizer = tokenizer,
        batch_size = 2,
        max_len = 256,
        num_workers = 0,
        device=device,
    )
    logger.info(f"Train loader:")
    for x, y in train_dataloader:
        logger.info(f"x.shape: {x.shape}, y.shape: {y.shape}")
    
    logger.info("Validation loader:")
    for x, y in valid_dataloader:
        logger.info(f"x.shape: {x.shape}, y.shape: {y.shape}")
    
    # train, validation tokens
    train_tokens = 0
    for input_batch, target_batch in train_dataloader:
        train_tokens += input_batch.numel()
    val_tokens = 0
    for input_batch, target_batch in valid_dataloader:
        val_tokens += input_batch.numel()
    logger.info(f"Training tokens: {train_tokens}")
    logger.info(f"Validation tokens: {val_tokens}")
    logger.info(f"All tokens: {train_tokens + val_tokens}")

    # model
    model = Model(GPT2_CONFIG).to(device)
    
    # loss
    with torch.no_grad():
        train_loss = calc_loss_loader("pretrain", train_dataloader, model, device)
        valid_loss = calc_loss_loader("pretrain", valid_dataloader, model, device)
    logger.info(f"Training loss: {train_loss}")
    logger.info(f"Validation loss: {valid_loss}")




# 测试代码 main 函数
def main():
    from models.gpt2_model_cfg.model_cfgs import device, tokenizer, GPT2_124M_CONFIG

    # text_generation(GPT2_124M_CONFIG, device)

    GPT2_124M_CONFIG.context_length = 6
    test_training(tokenizer, GPT2_124M_CONFIG, device)

if __name__ == "__main__":
    main()
