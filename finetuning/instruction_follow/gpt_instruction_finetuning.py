# -*- coding: utf-8 -*-

# ***************************************************
# * File        : gpt_finetuning_instruction.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2025-02-16
# * Version     : 0.1.021622
# * Description : supervised instruction finetuning
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import os
import sys
ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
import re
import json
import time
from tqdm import tqdm

import tiktoken
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# data
from finetuning.instruction_follow.data_load import load_file
from finetuning.instruction_follow.data_loader import create_dataloader
from finetuning.instruction_format import format_input_alpaca
# tokenizer
from tokenizer.tokenization import text_to_token_ids, token_ids_to_text
# model
from models.gpt import Model
from training.gpt_generate import generate
from models_load.openai_gpt2_weights_load_hf import load_weights
# model training
from training.calc_loss import _calc_loss_batch, _calc_loss_loader
from training.generate import _generate_and_print_sample
from training.train_funcs import _select_optimizer
from training.plot_losses import _plot_losses
from training.save_load_model import _save_model
# tools
from utils.argsparser_tools import DotDict
from utils.device import device
from utils.log_util import logger

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]

# ------------------------------
# data
# ------------------------------
def _build_data(data_path: str, train_ratio: float = 0.85, test_ratio: float = 0.10, batch_size: int = 8):
    """
    create dataset and dataloader
    """
    # data
    data = load_file(file_path = data_path)
    logger.info(f"Number of entries: {len(data)}")
    # data split ratio
    train_portion = int(len(data) * train_ratio)
    test_portion = int(len(data) * test_ratio)
    valid_portion = len(data) - train_portion - test_portion
    # data split
    train_data = data[:train_portion]
    test_data = data[train_portion:(train_portion + test_portion)]
    valid_data = data[(train_portion + test_portion):]
    # logger.info(f"train_data: \n{train_data[0]}")
    # logger.info(f"test_data: \n{test_data[0]}")
    # logger.info(f"valid_data: \n{valid_data[0]}")
    logger.info(f"Training data length: {len(train_data)}")
    logger.info(f"Test data length: {len(test_data)}")
    logger.info(f"Validation data length: {len(valid_data)}")
    # dataset and dataloader
    train_dataset, train_dataloader = create_dataloader(
        data = train_data,
        batch_size = batch_size,
        shuffle = True,
        drop_last = True,
    )
    test_dataset, test_dataloader = create_dataloader(
        data = test_data,
        batch_size = batch_size,
        shuffle = False,
        drop_last = False,
    )
    valid_dataset, valid_dataloader = create_dataloader(
        data = valid_data,
        batch_size = batch_size,
        shuffle = False,
        drop_last = False,
    )

    return (
        train_data, train_dataset, train_dataloader, 
        test_data, test_dataset, test_dataloader,
        valid_data, valid_dataset, valid_dataloader,
    )


# ------------------------------
# model, optimizer, loss
# ------------------------------
def _build_model():
    """
    initializing a model with pretrained weights
    """
    # Loading pretrained LLM   
    choose_model = "gpt2-medium (355M)"

    # model base cofig
    base_config = {
        "vocab_size": 50257,     # Vocabulary size
        "context_length": 1024,  # Context length
        "dropout": 0.0,          # Dropout rate
        "qkv_bias": True         # Query-key-value bias
    }
    # huggingface allowed model names  
    gpt2_model_configs = {
        "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
        "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
        "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
        "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
    }
    base_config.update(gpt2_model_configs[choose_model])
    base_config = DotDict(base_config)

    # huggingface gpt2 model
    from transformers import GPT2Model
    gpt2_hf_model_names = {
        "gpt2-small (124M)": "openai-community/gpt2",
        "gpt2-medium (355M)": "openai-community/gpt2-medium",
        "gpt2-large (774M)": "openai-community/gpt2-large",
        "gpt2-xl (1558M)": "openai-community/gpt2-xl"
    }
    gpt2_hf = GPT2Model.from_pretrained(
        gpt2_hf_model_names[choose_model],
        cache_dir = "./downloaded_models/gpt2_model"
    )
    gpt2_hf.eval();

    # custom gpt model
    model = Model(base_config)
    load_weights(model, gpt2_hf, base_config)
    model.eval(); 

    return model, base_config, choose_model


# ------------------------------
# Finetuning LLM on instruction data
# ------------------------------
def valid(model, train_loader, val_loader, device, eval_iter):
    """
    model evaluate
    """
    model.eval()
    with torch.no_grad():
        train_loss = _calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = _calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    
    return train_loss, val_loss


def _train_model_simple(model, 
                        train_loader, 
                        val_loader, 
                        optimizer, 
                        device, 
                        train_epochs,
                        eval_freq, 
                        eval_iter, 
                        start_context):
    # Initialize lists to track losses and tokens seen
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1
    # Main training loop
    for epoch in range(train_epochs):
        # Set model to training mode
        model.train()
        for input_batch, target_batch in train_loader:
            # Reset loss gradients from previous batch iteration
            optimizer.zero_grad()
            loss = _calc_loss_batch(input_batch, target_batch, model, device)
            # Calculate loss gradients
            loss.backward()  
            # Update model weights using loss gradients
            optimizer.step()

            tokens_seen += input_batch.numel()
            global_step += 1

            # Optional evaluation step
            if global_step % eval_freq == 0:
                train_loss, val_loss = valid(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                logger.info(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        # Print a sample text after each epoch
        _generate_and_print_sample(model, device, start_context)

    return train_losses, val_losses, track_tokens_seen


def train(model, train_dataloader, valid_dataloader, device, valid_data):
    """
    model training
    """
    # training start time
    training_start_time = time.time()
    # seed
    torch.manual_seed(123)
    # training epochs
    train_epochs = 2
    # move model to device
    model.to(device)
    # optimizer
    optimizer = _select_optimizer(model=model)
    # model training
    train_losses, val_losses, tokens_seen = _train_model_simple(
        model, 
        train_dataloader, 
        valid_dataloader, 
        optimizer, 
        device,
        train_epochs=train_epochs, 
        eval_freq=5, 
        eval_iter=5,
        start_context = format_input_alpaca(valid_data[0]), 
    )
    
    # training end time
    training_end_time = time.time()
    # training time
    execution_time_minutes = (training_end_time - training_start_time) / 60
    logger.info(f"Training completed in {execution_time_minutes:.2f} minutes.")

    # plot losses
    epochs_tensor = torch.linspace(0, train_epochs, len(train_losses))
    _plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)


def _extract_save_responses(test_data, model, base_config):
    """
    Extracting and saving responses
    """
    torch.manual_seed(123)
    for entry in test_data[:3]:
        input_text = format_input_alpaca(entry)
        token_ids = generate(
            model=model,
            token_idx=text_to_token_ids(input_text).to(device),
            max_new_tokens=256,
            context_size=base_config.context_length,
            eos_id=50256
        )
        generated_text = token_ids_to_text(token_ids)
        response_text = (
            generated_text[len(input_text):]
            .replace("### Response:", "")
            .strip()
        )
        logger.info(input_text)
        logger.info(f"\nCorrect response:\n>> {entry['output']}")
        logger.info(f"\nModel response:\n>> {response_text.strip()}")
        logger.info("-------------------------------------")


    for i, entry in tqdm(enumerate(test_data), total=len(test_data)):
        input_text = format_input_alpaca(entry)
        token_ids = generate(
            model=model,
            token_idx=text_to_token_ids(input_text).to(device),
            max_new_tokens=256,
            context_size=base_config.context_length,
            eos_id=50256
        )
        generated_text = token_ids_to_text(token_ids)
        response_text = generated_text[len(input_text):].replace("### Response:", "").strip()
        test_data[i]["model_response"] = response_text


def _build_test_data(test_data):
    """
    build instruction data with response
    """
    result_path = "./saved_results/test_results/"
    os.makedirs(result_path, exist_ok=True)
    with open(os.path.join(result_path, "instruction-data-with-response.json"), "w") as file:
        json.dump(test_data, file, indent=4)



# 测试代码 main 函数
def main():
    data_path = "./dataset/finetuning/instruction-data.json"
    pretrained_model_path = "./saved_results/finetuning_pretrained_models/"
    os.makedirs(pretrained_model_path, exist_ok=True)
    train_ratio = 0.85
    test_ratio = 0.10
    batch_size = 8
    max_new_tokens = 35
    # ------------------------------
    # data
    # ------------------------------
    (
        train_data, train_dataset, train_dataloader, 
        test_data, test_dataset, test_dataloader,
        valid_data, valid_dataset, valid_dataloader,
    ) = _build_data(data_path, train_ratio, test_ratio, batch_size)

    # ------------------------------
    # model
    # ------------------------------
    model, base_config, choose_model = _build_model()
    """
    # model test
    torch.manual_seed(123)
    input_text = format_input_alpaca(valid_data[0])
    logger.info(f"input text: \n{input_text}")

    token_ids = generate(
        model = model, 
        token_idx = text_to_token_ids(input_text),
        max_new_tokens = max_new_tokens,
        context_size = base_config.context_length,
        eos_id = 50256,
    )
    generated_text = token_ids_to_text(token_ids)
    logger.info(f"generated text: \n{generated_text}")
    
    response_text = (
        generated_text[len(input_text):]
        .replace("### Response:", "")
        .strip()
    )
    logger.info(f"response text: \n{response_text}")
    """
    train(model, train_dataloader, valid_dataloader, device, valid_data)
    _extract_save_responses(test_data, model, base_config)
    _build_test_data(test_data)
    _save_model(model, pretrained_model_path, choose_model)


if __name__ == "__main__":
    main()
