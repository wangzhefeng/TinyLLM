# -*- coding: utf-8 -*-

# ***************************************************
# * File        : nanoVLM.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-07-08
# * Version     : 1.0.070814
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
import argparse
import warnings
warnings.filterwarnings("ignore")

import torch
from nanoVLM.util import (
    is_master, is_dist, 
    init_dist, destory_dist,
    seed_torch, seed_worker
)

# set torch seed
seed_torch()

from nanoVLM.config import VLMConfig, TrainConfig

# Otherwise, the tokenizer will throw a warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# TODO local log
LOGGING_LABEL = Path(__file__).name[:-3]
os.environ['LOG_NAME'] = LOGGING_LABEL
from utils.log_util import logger


def train(train_cfg, vlm_cfg):
    pass




# 测试代码 main 函数
def main():
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr_mp', type=float, help='Learning rate for the mapping network')
    parser.add_argument('--lr_backbones', type=float, help='Learning rate for the backbones')
    parser.add_argument('--vlm_checkpoint_path', type=str, help='Path to the VLM checkpoint for loading or saving')
    parser.add_argument('--compile', type=bool, help='Use torch.compile to optimize the model')
    parser.add_argument('--log_wandb', type=bool, help='Log to wandb')
    parser.add_argument('--resume_from_vlm_checkpoint', type=bool, default=False, help='Resume training from VLM checkpoint specified by vlm_checkpoint_path (or default if not provided)')
    parser.add_argument('--no_log_wandb', action='store_true', help='Do not log to wandb')

    args = parser.parse_args()

    # model configs
    vlm_cfg = VLMConfig()
    logger.info(f"vlm config: \n{vlm_cfg}")

    # model training configs
    train_cfg = TrainConfig()
    if args.lr_mp is not None:
        train_cfg.lr_mp = args.lr_mp
    if args.lr_backbones is not None:
        train_cfg.lr_backbones = args.lr_backbones
    if args.vlm_checkpoint_path is not None:
        vlm_cfg.vlm_checkpoint_path = args.vlm_checkpoint_path
    if args.compile is not None:
        train_cfg.compile = args.compile
    if args.no_log_wandb is True:
        train_cfg.log_wandb = False
    if args.resume_from_vlm_checkpoint and args.vlm_checkpoint_path is not None:
        train_cfg.resume_from_vlm_checkpoint = True
        # When resuming a full VLM, we don't need to load individual backbone weights from original sources
        vlm_cfg.vlm_load_backbone_weights = False
    logger.info(f"train config: \n{train_cfg}")
    
    # initiation distributed training
    if is_master():
        print("--- VLM Config ---")
        print(vlm_cfg)
        print("--- Train Config ---")
        print(train_cfg)
    
    # model training
    # train(train_cfg, vlm_cfg)

    # destory distributed training
    if is_dist():
        destory_dist()

if __name__ == "__main__":
    main()
