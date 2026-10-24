# -*- coding: utf-8 -*-

# ***************************************************
# * File        : exp_train_rm.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2026-10-24
# * Version     : 1.0.102421
# * Description : Reward Model 训练实验（RLHF 一阶段）：
# *               复用 DPO 的偏好对数据（data_provider/dpo），
# *               backbone + v_head 用 Bradley-Terry 成对损失训练；
# *               训练后的 RM 可作为 PPO（model_rlhf/ppo/main.py）的
# *               learned reward 来源。
# * Requirement : torch
# ***************************************************

__all__ = ["ModelTrainReward"]

# python libraries
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)

import torch

# data（复用 DPO 偏好对）
from post_training.preference.data_provider.dpo import data_loader
# model
from models.gpt2.gpt2_124M import Model
from post_training.preference.rm import RewardModel, rm_pairwise_step
# tokenizer
from layers.tokenizers.tokenization import choose_tokenizer
# utils
from utils.optimizer import select_optimizer
from utils.device import device_setting
from utils.log_util import logger

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


class ModelTrainReward:
    """在偏好对数据上训练 Reward Model 的实验类。"""

    def __init__(self, args):
        self.args = args
        self.device = device_setting()
        self.tokenizer = choose_tokenizer(tokenizer_model = self.args.tokenizer_model)
        self.pad_token_id = self.tokenizer.encode(
            "<|endoftext|>", allowed_special = {"<|endoftext|>"})[0]

    def _get_data(self):
        train_data, _, _ = data_loader.load_split_data(
            self.args.data_path,
            self.args.train_ratio,
            self.args.test_ratio,
        )
        _, train_dataloader = data_loader.create_dataloader(
            data = train_data,
            device = self.device,
            tokenizer = self.tokenizer,
            pad_token_id = self.pad_token_id,
            mask_prompt_tokens = True,
            allowed_max_length = self.args.context_length,
            batch_size = self.args.batch_size,
            shuffle = True,
            drop_last = True,
        )
        return train_dataloader

    def _get_model(self):
        model = Model(self.args)
        rm = RewardModel(model, self.args).to(self.device)
        optimizer = select_optimizer(
            rm, learning_rate = self.args.learning_rate, weight_decay = 0.0)
        return rm, optimizer

    def train(self):
        train_dataloader = self._get_data()
        rm, optimizer = self._get_model()
        rm.train()
        for epoch in range(self.args.num_epochs):
            epoch_loss, epoch_margin, n_batch = 0.0, 0.0, 0
            for batch in train_dataloader:
                loss, margin = rm_pairwise_step(
                    rm, optimizer,
                    chosen_ids = batch["chosen"],
                    rejected_ids = batch["rejected"],
                    pad_token_id = self.pad_token_id,
                )
                epoch_loss += loss
                epoch_margin += margin
                n_batch += 1
            if n_batch:
                logger.info(
                    f"Epoch {epoch + 1}/{self.args.num_epochs} "
                    f"rm_loss: {epoch_loss / n_batch:.4f} "
                    f"reward_margin: {epoch_margin / n_batch:.4f}")
        # 保存 backbone + v_head
        torch.save(rm.state_dict(), self.args.save_path)
        logger.info(f"Reward model saved to {self.args.save_path}")
        return rm


# 测试代码 main 函数
def main():
    from utils.args_tools import DotDict

    from post_training.preference.data_provider.dpo import data_config

    args = DotDict({
        "vocab_size": 50257,
        "context_length": 1024,
        "embed_dim": 768,
        "n_layers": 12,
        "n_heads": 12,
        "dropout": 0.0,
        "qkv_bias": True,
        "dtype": torch.float32,
        "tokenizer_model": "tiktoken_gpt2_bpe",
        "data_path": data_config.instruction_data_with_preference_path,
        "train_ratio": 0.85,
        "test_ratio": 0.10,
        "batch_size": 2,
        "num_epochs": 1,
        "learning_rate": 5e-5,
        "save_path": "./results/checkpoints/rm/gpt2_rm.pt",
    })
    ModelTrainReward(args).train()

if __name__ == "__main__":
    main()
