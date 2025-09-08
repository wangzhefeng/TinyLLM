# -*- coding: utf-8 -*-

# ***************************************************
# * File        : hparam_search.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-09-07
# * Version     : 1.0.090700
# * Description : description
# * Link        : https://github.com/rasbt/LLMs-from-scratch/blob/main/ch05/05_bonus_hparam_tuning/hparam_search.py
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
import math
import itertools
import warnings
warnings.filterwarnings("ignore")

import torch

from layers.tokenizers.tokenization import choose_tokenizer
from utils.device import device_setting

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]
os.environ['LOG_NAME'] = LOGGING_LABEL
from utils.log_util import logger


# Define a grid of hyperparameters to search over
HPARAM_GRID = {
    "batch_size": [2, 4, 8, 16],
    "dropout": [0.0, 0.1, 0.2],
    "warmup_iters": [10, 20, 30],
    "weight_decay": [0.0, 0.1, 0.01],
    "peak_lr": [0.001, 0.005, 0.0001, 0.0005],
    "initial_lr": [0.0001, 0.00005],
    "min_lr": [0.0001, 0.00001, 0.00001],
    "train_epochs": [5, 10, 15, 20, 25],
}




# 测试代码 main 函数
def main():
    # random seed
    torch.manual_seed(123)

    # device
    device = device_setting(verbose=True)

    # tokenizer
    tokenizer = choose_tokenizer(tokenizer_model="tiktoken_gpt2_bpe")

    # Generate all combinations of hyperparameters
    hyperparameter_combinations = list(itertools.product(*HPARAM_GRID.values()))
    total_combinations = len(hyperparameter_combinations)
    logger.info(f"Total hyperparameter configurations: {total_combinations}")

    # Placeholder for the best loss and best hyperparameters
    best_val_loss = float("inf")
    best_hparams = {}
    interrupted = False
    current_config = 0
    for combination in hyperparameter_combinations:
        try:
            current_config += 1
            logger.info(f"Evaluating configuration {current_config} of {total_combinations}")
            
            # Unpack the current combination of hyperparameters
            HPARAM_CONFIG = dict(zip(HPARAM_GRID.keys(), combination))

            # data loader
            train_loader, val_loader = None, None

            # model training
            train_loss, val_loss = None, None

            # Log the best hyperparameters based on validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_train_loss = train_loss
                best_hparams = HPARAM_CONFIG
        except KeyboardInterrupt:
            logger.info(f"Hyperparameter search completed.")
            logger.info(f"Best hyperparameters: {best_hparams}")
            logger.info(f"Best Val loss: {best_val_loss} | Training loss {train_loss}")
            interrupted = True
            break
    
    if not interrupted:
        logger.info(f"Hyperparameter search completed.")
        logger.info(f"Best hyperparameters: {best_hparams}")
        logger.info(f"Best Val loss: {best_val_loss} | Training loss {train_loss}")

if __name__ == "__main__":
    main()
