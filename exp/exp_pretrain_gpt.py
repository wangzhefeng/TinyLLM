# -*- coding: utf-8 -*-

# ***************************************************
# * File        : run.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-01-23
# * Version     : 1.0.012322
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
import time
import warnings

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import torch
import torch.nn as nn
import torch.nn.functional as F

# data
from data_provider.data_load_pretrain import data_load
from data_provider.data_loader import create_dataloader
# tokenizer
from layers.tokenization import text_to_token_ids, token_ids_to_text
# model
from exp.exp_basic import Exp_Basic
from models.gpt_generate import generate_text_simple, generate
from training.train_funcs import adjust_learning_rate, EarlyStopping
from training.save_load_model import load_model_weights
# utils
# from utils.device import device
from utils.log_util import logger

warnings.filterwarnings("ignore")

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


class Model_Pretrain(Exp_Basic):
    
    def __init__(self, args):
        super(Model_Pretrain, self).__init__(args)
    
    def _build_data(self):
        """
        build dataset and dataloader
        """
        # data load
        raw_text = data_load(url=self.args.data_source)
        # dataset
        split_idx = int(self.args.train_ratio * len(raw_text))
        train_data = raw_text[:split_idx]
        val_data = raw_text[split_idx:]
        # dataloader
        train_loader = create_dataloader(
            train_data,
            batch_size=self.args.batch_size,
            max_length=self.args.context_length,
            stride=self.args.context_length,
            drop_last=True,
            shuffle=True,
            num_workers=0,
        )
        val_loader = create_dataloader(
            val_data,
            batch_size=self.args.batch_size,
            max_length=self.args.context_length,
            stride=self.args.context_length,
            drop_last=False,
            shuffle=False,
            num_workers=0,
        )
        
        return train_loader, val_loader

    def _build_model(self):
        """
        build model
        """
        # model instance
        model = self.model_dict[self.args.model_name].Model(self.args).float()
        # 单机多卡训练
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        # 打印模型参数量
        total = sum([param.nelement() for param in model.parameters()])
        logger.info(f'Number of parameters: {(total / 1e6):.2f}M')
        
        return model

    def _select_optimizer(self):
        """
        optimizer
        """
        optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr = self.args.learning_rate, 
            weight_decay = self.args.weight_decay
        )

        return optimizer

    def _select_criterion(self):
        """
        loss
        """
        criterion = nn.CrossEntropyLoss()

        return criterion

    def _calc_loss_batch(self, model, input_batch, target_batch):
        # criterion
        criterion = self._select_criterion()
        # training data batch
        input_batch = input_batch.to(self.device)
        target_batch = target_batch.to(self.device)
        # forward
        logits = model(input_batch)
        # loss
        loss = criterion(logits.flatten(0, 1), target_batch.flatten())

        return loss

    def _calc_loss_loader(self, model, data_loader, num_batches = None):
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
                loss = self._calc_loss_batch(model, input_batch, target_batch)
                total_loss += loss.item()
            else:
                break
        
        return total_loss / num_batches

    def _get_model_path(self, setting):
        """
        模型保存路径
        """
        # 模型保存路径
        model_path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(model_path, exist_ok=True)
        # 最优模型保存路径
        best_model_path = f"{model_path}/checkpoint.pth"
        
        return best_model_path

    def _get_test_results_path(self, setting):
        """
        结果保存路径
        """
        results_path = os.path.join(self.args.test_results, setting + "/")
        os.makedirs(results_path, exist_ok=True)
        
        return results_path

    def train(self, setting, eval_freq: int = 2, eval_iter: int = 2, 
              start_context: str = "Every effort moves you"):
        # build dataloader
        train_loader, val_loader = self._build_data()
        # checkpoint path
        best_model_path = self._get_model_path(setting)
        # early stopping
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        # training start time
        training_start_time = time.time()        
        # train steps
        train_steps = len(train_loader)
        logger.info(f"train_steps: {train_steps}")
        # train optimizer
        optimizer = self._select_optimizer()
        # initialize list to track losses and tokens seen
        train_losses, val_losses = [], []
        track_tokens_seen = []
        tokens_seen = 0
        global_step = -1
        # main training loop
        for epoch in range(self.args.train_epochs):
            # time: epoch 模型训练开始时间
            epoch_start_time = time.time()
            # iter count
            iter_count = 0
            # training mode
            self.model.train()
            # model training
            for i, (input_batch, target_batch) in enumerate(train_loader):
                # update iter count
                iter_count += 1
                # forward
                optimizer.zero_grad()
                loss = self._calc_loss_batch(self.model, input_batch, target_batch)
                # backward
                loss.backward()
                optimizer.step()
                # update tokens seen
                tokens_seen += input_batch.numel()
                # update global step
                global_step += 1
                # optional evaluation step
                if global_step % eval_freq == 0:
                    train_loss, val_loss = self.vali(train_loader, val_loader, eval_iter)
                    # collect losses and tokens seen
                    train_losses.append(train_loss)
                    val_losses.append(val_loss)
                    track_tokens_seen.append(tokens_seen)
                    logger.info(f"Epoch {epoch + 1} (Step {global_step:06d}): Train loss: {train_loss:.3f}, Val loss {val_loss:.3f}")
                    # calculate training left time
                    speed = (time.time() - training_start_time) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    logger.info(f'Epoch: {epoch + 1}, \tSpeed: {speed:.4f}s/iter; left time: {left_time:.4f}s')
                    iter_count = 0
                    training_start_time = time.time()
            # 日志打印: 训练 epoch、每个 epoch 训练的用时
            logger.info(f"Epoch: {epoch + 1}, Cost time: {time.time() - epoch_start_time}")
            # 早停机制、模型保存
            early_stopping(val_loss, self.model, best_model_path)
            if early_stopping.early_stop:
                logger.info(f"Epoch: {epoch + 1}, Early stopping...")
                break
            # 学习率调整
            adjust_learning_rate(optimizer, epoch + 1, self.args)
            
            # print a sample text after each epoch
            self.generate_and_print_sample(self.model, start_context)
        
        # calc training time
        training_end_time = time.time()
        execution_time_minutes = (training_end_time - training_start_time) / 60
        logger.info(f"Training Completed in {execution_time_minutes:.2f} minutes.")

        # loss visual
        # self.plot_losses(self.args.train_epochs, tokens_seen, train_losses, val_losses)

        # ------------------------------
        # 模型加载
        # ------------------------------
        logger.info("Loading best model...")
        # self.model.load_state_dict(torch.load(best_model_path))
        load_model_weights(args = self.args, model_path=best_model_path, device=self.device)
        
        return train_losses, val_losses, track_tokens_seen
 
    def vali(self, train_loader, val_loader, eval_iter):
        """
        model evaluation
        """
        # inference mode
        self.model.eval()
        # model evaluation
        with torch.no_grad():
            train_loss = self._calc_loss_loader(self.model, train_loader, num_batches = eval_iter)
            val_loss = self._calc_loss_loader(self.model, val_loader, num_batches = eval_iter)
        self.model.train()
        
        return train_loss, val_loss
    
    # TODO
    def test(self):
        pass
    
    def generate_and_print_sample(self, model, start_context):
        model.eval()
        context_size = self.model.pos_emb.weight.shape[0]
        encoded = text_to_token_ids(start_context).to(self.device)
        with torch.no_grad():
            token_ids = generate(
                model = model, 
                token_idx = encoded,
                max_new_tokens = self.args.max_new_tokens,
                context_size = context_size,
            )
            decoded_text = token_ids_to_text(token_ids)
            logger.info(decoded_text.replace('\n', ' '))
        model.train()

    def plot_losses(self, train_epochs, tokens_seen, train_losses, val_losses):
        epochs_seen = torch.linspace(0, train_epochs, len(train_losses))
        
        fig, ax1 = plt.subplots(figsize = (5, 3))
        # plot training and validation loss against epochs
        ax1.plot(epochs_seen, train_losses, label = "Training loss")
        ax1.plot(epochs_seen, val_losses, linestyle = "-.", label = "Validation loss")
        ax1.set_xlabel("Epochs")
        ax1.set_ylabel("Loss")
        ax1.legend(loc = "upper right")
        ax1.xaxis.set_major_locator(MaxNLocator(integer = True))
        # create a second x-axis for tokens seen
        ax2 = ax1.twiny()
        ax2.plot(tokens_seen, train_losses, alpha = 0)
        ax2.set_xlabel("Tokens seen")

        # adjust layout to make room
        fig.tight_layout()
        # plt.savefig("loss_plot.pdf")
        plt.show()




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
