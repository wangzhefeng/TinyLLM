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
ROOT = str(os.getcwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)
import math
import time
import warnings

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import torch
import torch.nn as nn

# data
from data_provider.pretrain.data_load import data_load
from data_provider.pretrain.data_loader import create_dataloader
# tokenizer
from tokenizer.tokenization import text_to_token_ids, token_ids_to_text
# model
from exp.exp_basic import Exp_Basic
from utils.train_utils.gpt_generate import generate
from utils.train_utils.train_funcs import adjust_learning_rate, EarlyStopping
# utils
from utils.log_util import logger

warnings.filterwarnings("ignore")

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


class Model_Pretrain(Exp_Basic):
    
    def __init__(self, args):
        super(Model_Pretrain, self).__init__(args)
    
    def _get_data(self):
        """
        build dataset and dataloader
        """
        # data load
        raw_text = data_load(url=self.args.data_source)
        # dataset
        split_idx = int(self.args.train_ratio * len(raw_text))
        train_data = raw_text[:split_idx]
        valid_data = raw_text[split_idx:]
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
        valid_loader = create_dataloader(
            valid_data,
            batch_size=self.args.batch_size,
            max_length=self.args.context_length,
            stride=self.args.context_length,
            drop_last=False,
            shuffle=False,
            num_workers=0,
        )
        
        return train_loader, valid_loader

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
        total = sum([param.numel() for param in model.parameters()])
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

    def _get_results_path(self, setting, training_iter):
        """
        结果保存路径
        """
        results_path = os.path.join(self.args.test_results, setting, str(training_iter))
        os.makedirs(results_path, exist_ok=True)
        
        return results_path

    def _plot_losses(self, tokens_seen, train_losses, valid_losses, results_path):
        # epochs seen
        epochs_seen = torch.linspace(0, self.args.train_epochs, len(train_losses))
        # Plot training and validation loss against epochs
        fig, ax1 = plt.subplots(figsize = (5, 3))
        ax1.plot(epochs_seen, train_losses, label = "Training loss")
        ax1.plot(epochs_seen, valid_losses, linestyle = "-.", label = "Validation loss")
        ax1.set_xlabel("Epochs")
        ax1.set_ylabel("Loss")
        ax1.legend(loc = "upper right")
        ax1.xaxis.set_major_locator(MaxNLocator(integer = True))
        # Create a second x-axis for tokens seen
        ax2 = ax1.twiny()
        ax2.plot(tokens_seen, train_losses, alpha = 0)
        ax2.set_xlabel("Tokens seen")
        # Aesthetic settings
        fig.tight_layout()
        # Fig save
        plt.savefig(os.path.join(results_path, "loss_plot.pdf"))
        # Fig show
        plt.show()

    def train(self, 
              training_iter, 
              setting, 
              eval_freq: int = 5, 
              eval_iter: int = 1, 
              start_context: str = "Every effort moves you"):
        # build dataloader
        train_loader, valid_loader = self._get_data()
        # train steps
        train_steps = len(train_loader)
        # checkpoint path
        best_model_path = self._get_model_path(setting)
        # test results path
        results_path = self._get_results_path(setting, training_iter)
        # early stopping
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True) 
        # train optimizer
        optimizer = self._select_optimizer()
        # mix precision
        """
        if self.args.use_amp:
            scaler = torch.amp.GradScaler(device = self.device)
        """
        # initialize list to track losses 
        train_losses, valid_losses = [], []
        # initialize list to track tokens seen
        track_tokens_seen = []
        # learning rate
        track_lrs = []
        # initialize tokens seen
        tokens_seen = 0
        # TODO initialize global steps
        global_step = -1
        # TODO 从优化器中获取最大学习率
        # peak_lr = optimizer.param_groups[0]["lr"]
        peak_lr = 0.001
        # 计算训练过程中总的迭代次数
        total_training_steps = len(train_loader) * self.args.train_epochs
       # warmup steps
        warmup_steps = int(0.2 * total_training_steps) 
        # 计算warmup阶段的迭代次数
        lr_increment = (peak_lr - self.args.initial_lr) / warmup_steps
        # training start time
        training_start_time = time.time()
        # main training loop
        for epoch in range(self.args.train_epochs):
            # epoch 模型训练开始时间
            epoch_start_time = time.time()
            # batch iter count
            iter_count = 0
            # training mode
            self.model.train()
            
            # model training
            for batch, (input_batch, target_batch) in enumerate(train_loader):
                # update global step
                global_step += 1
                # update batch iter count
                iter_count += 1
                
                # learning rate warmup
                # ------------------------
                # reset loss gradients from previous batch iteration
                optimizer.zero_grad()
                # 根据当前阶段（预热或余弦衰减）调整学习率
                if global_step < warmup_steps:
                    # 线性预热
                    lr = self.args.initial_lr + global_step * lr_increment
                else:
                    # 预热后余弦衰减
                    progress = (global_step - warmup_steps) / (total_training_steps - warmup_steps)
                    lr = self.args.min_lr + (peak_lr - self.args.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
                # 将计算出的学习率应用到优化器中
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr
                # 记录当前学习率
                track_lrs.append(lr)
                
                # forward
                # ------------------------
                # calculate train loss
                loss = self._calc_loss_batch(self.model, input_batch, target_batch)
                """
                if self.args.use_amp:
                    with torch.amp.autocast(device_type=self.args.gpu_type):
                        loss = self._calc_loss_batch(self.model, input_batch, target_batch)
                else:
                    loss = self._calc_loss_batch(self.model, input_batch, target_batch)
                """
                # backward
                # ------------------------
                # calculate loss gradient
                loss.backward()
                # 在预热阶段后应用梯度裁剪，防止梯度爆炸
                if global_step > warmup_steps:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm = 1.0) 
                # update model weights using loss gradients
                optimizer.step() 
                """
                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # calculate loss gradient
                    loss.backward()
                    # update model weights using loss gradients
                    optimizer.step()
                """
                # collect data
                # ------------------------
                # update tokens seen
                tokens_seen += input_batch.numel() 
                # optional evaluation step
                # ------------------------
                if global_step % eval_freq == 0:
                    train_loss, valid_loss = self.valid(train_loader, valid_loader, eval_iter)
                    # collect losses and tokens seen
                    train_losses.append(train_loss)
                    valid_losses.append(valid_loss)
                    track_tokens_seen.append(tokens_seen)
                    logger.info(f"\t\tEpoch {epoch + 1} Batch {batch + 1} \t(Step {global_step:06d}): Train loss: {train_loss:.3f}, Val loss {valid_loss:.3f}")

                    # calculate training left time
                    speed = (time.time() - training_start_time) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - batch)
                    # logger.info(f'\t\tEpoch {epoch + 1} Batch {batch + 1} \tSpeed: {speed:.4f}s/iter; left time: {left_time:.4f}s')
                    
                    iter_count = 0
                    training_start_time = time.time()
            
            # 早停机制、模型保存
            early_stopping(epoch = epoch + 1, val_loss = valid_loss, model = self.model, path = best_model_path)
            if early_stopping.early_stop:
                logger.info(f"\t\tEpoch {epoch + 1} \t\tEarly stopping...")
                break
            # 学习率调整
            adjust_learning_rate(optimizer = optimizer, epoch = epoch + 1, args = self.args)
            # print a sample text after each epoch
            self.inference(epoch = epoch + 1, start_context = start_context)
            
            # calculate one epoch training used time
            logger.info(f"\t\tEpoch {epoch + 1} \t\tcost time: {time.time() - epoch_start_time}s.")
        
        # calculate all epoch training used time
        logger.info(f"\t\tTraining Iter {training_iter + 1} \tcost time: {((time.time() - training_start_time) / 60):.2f}mins.")

        # loss visual
        self._plot_losses(track_tokens_seen, train_losses, valid_losses, results_path)

        # model load
        logger.info("\t\tLoading best model...")
        self.model.load_state_dict(torch.load(best_model_path))
        
        return train_losses, valid_losses, track_tokens_seen
 
    def valid(self, train_loader, valid_loader, eval_iter):
        """
        model evaluation
        """
        # inference mode
        self.model.eval()
        # model evaluation
        with torch.no_grad():
            train_loss = self._calc_loss_loader(
                self.model, 
                train_loader, 
                num_batches = eval_iter
            )
            valid_loss = self._calc_loss_loader(
                self.model, 
                valid_loader, 
                num_batches = eval_iter
            )
        # training mode
        self.model.train()
        
        return train_loss, valid_loss
    
    def inference(self, epoch, start_context):
        # inference mode
        self.model.eval()
        # context size
        context_size = self.model.pos_emb.weight.shape[0]
        # start context tokenization
        encoded = text_to_token_ids(start_context).to(self.device)
        # generate text
        with torch.no_grad():
            token_ids = generate(
                model = self.model, 
                token_idx = encoded,
                max_new_tokens = self.args.max_new_tokens,
                context_size = context_size,
            )
            decoded_text = token_ids_to_text(token_ids).replace("\n", " ")
            logger.info(f"\t\tEpoch {epoch} \t\tdeocde_text: {decoded_text}")
        # train mode
        self.model.train()
 



# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
