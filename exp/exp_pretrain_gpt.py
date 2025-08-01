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
# ***************************************************

# python libraries
import os
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

# data
from data_provider.data_loader import create_dataloader
# tokenizer
from layers.tokenizers.tokenization import (
    choose_tokenizer,
    text_to_token_ids, 
    token_ids_to_text
)
# model
from exp.exp_basic import Exp_Basic
# training
from utils.llm.calc_loss import calc_loss_batch, calc_loss_loader
from utils.llm.train_funcs import select_optimizer
from utils.llm.train_funcs import adjust_learning_rate, EarlyStopping
from utils.llm.gpt_generate import generate
from utils.plot_losses import plot_losses
# utils
from utils.model_memory import model_memory_size
from utils.timestamp_utils import from_unix_time
from utils.log_util import logger

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


class Model_Pretrain(Exp_Basic):

    def __init__(self, args):
        logger.info(f"{41 * '-'}")
        logger.info("Initializing Experiment...")
        logger.info(f"{41 * '-'}")
        super(Model_Pretrain, self).__init__(args) 
    
    def _get_tokenizer(self):
        """
        get tokenizer
        """
        tokenizer = choose_tokenizer(tokenizer_model = self.args.tokenizer_model)
        
        return tokenizer
    
    def _get_data(self):
        """
        get dataset and dataloader
        """
        train_data, train_loader = create_dataloader(
            data_path=self.args.data_path,
            data_file=self.args.data_file,
            flag="train",
            train_ratio=self.args.train_ratio,
            tokenizer=self.tokenizer,
            batch_size=self.args.batch_size,
            max_len=self.args.context_length,
            stride=self.args.context_length,
            num_workers=self.args.num_workers,
        )
        valid_data, valid_loader = create_dataloader(
            data_path=self.args.data_path,
            data_file=self.args.data_file,
            flag="valid",
            train_ratio=self.args.train_ratio,
            tokenizer=self.tokenizer,
            batch_size=self.args.batch_size,
            max_len=self.args.context_length,
            stride=self.args.context_length,
            num_workers=self.args.num_workers,
        )
        
        return train_loader, valid_loader 

    def _build_model(self):
        """
        build model
        """
        # model instance
        logger.info(f"Initializing model {self.args.model_name}...")
        model = self.model_dict[self.args.model_name].Model(self.args)
        # 单机多卡训练
        if self.args.use_gpu and self.args.use_multi_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        elif self.args.use_ddp:
            model = DDP(model, device_ids=[self.device])
        # 打印模型参数量
        total_memory_gb = model_memory_size(model, verbose=True)
        
        return model

    def _get_model_path(self, setting):
        """
        模型保存路径
        """
        # 模型保存路径
        model_path = Path(self.args.checkpoints).joinpath(setting)
        os.makedirs(model_path, exist_ok=True)
        # 最优模型保存路径
        model_checkpoint_path = f"{model_path}/checkpoint.pth"
        
        return model_checkpoint_path

    def _get_results_path(self, setting):
        """
        结果保存路径
        """
        results_path = Path(self.args.test_results).joinpath(setting)
        os.makedirs(results_path, exist_ok=True)
        
        return results_path

    def _save_checkpoint(self, epoch, model_path):
        """
        模型 checkpoint 保存
        """
        ckp = self.model.module.state_dict()
        torch.save(ckp, model_path)


    def train(self, training_iter: int, setting: str, eval_freq: int=5, eval_iter: int=1):
        logger.info(f"{43 * '-'}")
        logger.info(f"Model start training...")
        logger.info(f"{43 * '-'}")
        # training start time
        train_start_time = time.time()
        logger.info(f"Train start time: {from_unix_time(train_start_time).strftime('%Y-%m-%d %H:%M:%S')}")
        # build dataloader
        train_loader, valid_loader = self._get_data()
        logger.info(f"Train and valid dataloader has builded...")
        # train steps
        train_steps = len(train_loader)
        logger.info(f"Train total steps: {train_steps}")
        # checkpoint path
        model_checkpoint_path = self._get_model_path(setting)
        logger.info(f"Train checkpoint will be saved in path: {model_checkpoint_path}")
        # test results path
        results_path = self._get_results_path(setting)
        logger.info(f"Train results will be saved in path: {results_path}") 
        # train optimizer
        optimizer = select_optimizer(
            self.model, 
            learning_rate=self.args.learning_rate, 
            weight_decay=self.args.weight_decay
        )
        logger.info(f"Train optimizer has builded...")
        # early stopping
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        logger.info(f"Train early stopping instance has builded, patience: {self.args.patience}")
        # auto mix precision
        if self.args.use_amp:
            scaler = torch.amp.GradScaler(device = self.device)
            logger.info(f"Train auto mix precision instance has builded...") 
        """
        # learning rate
        track_lrs = []
        # 从优化器中获取最大学习率
        peak_lr = 0.001  # peak_lr = optimizer.param_groups[0]["lr"]
        # 计算训练过程中总的迭代次数
        total_training_steps = len(train_loader) * self.args.train_epochs
        # warmup steps
        warmup_steps = int(0.2 * total_training_steps) 
        # 计算 warmup 阶段的迭代次数
        lr_increment = (peak_lr - self.args.initial_lr) / warmup_steps
        """
        # TODO initialize train iter global steps
        global_step = -1
        # initialize list to track train iter losses
        train_losses, valid_losses = [], []
        # initialize tokens seen
        tokens_seen = 0
        # initialize list to track train iter tokens seen
        track_tokens_seen = []
        # main training loop
        for epoch in range(self.args.train_epochs):
            # epoch 模型训练开始时间
            epoch_start_time = time.time()
            logger.info(f"\t\tEpoch {epoch + 1}: start time: {from_unix_time(epoch_start_time).strftime('%Y-%m-%d %H:%M:%S')}")
            
            # initialize epcoh iter count
            iter_count = 0
            # TODO initialize epoch train loss collector
            epoch_train_loss = []

            # training mode
            self.model.train()
            # ------------------------------
            # model training
            # ------------------------------
            for batch, (input_batch, target_batch) in enumerate(train_loader):
                # update train iter global step
                global_step += 1
                # update epoch iter count
                iter_count += 1
                # update tokens seen
                tokens_seen += input_batch.numel()
                
                # reset loss gradients from previous batch iteration
                optimizer.zero_grad()
                """
                # TODO learning rate warmup
                # ------------------------
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
                """
                
                # forward
                # ------------------------
                # calculate train loss
                if self.args.use_amp:
                    with torch.amp.autocast(device_type=self.args.gpu_type):
                        loss = calc_loss_batch(
                            task_name=self.args.task_name, 
                            input_batch=input_batch,
                            target_batch=target_batch,
                            model=self.model, 
                            device=self.device,
                        )
                else:
                    loss = calc_loss_batch(
                        task_name=self.args.task_name, 
                        input_batch=input_batch,
                        target_batch=target_batch,
                        model=self.model, 
                        device=self.device,
                    )
                # collect epoch loss value
                epoch_train_loss.append(loss.item())

                # backward
                # ------------------------
                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    # TODO 在预热阶段后应用梯度裁剪，防止梯度爆炸
                    # if global_step > warmup_steps:
                    #     torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm = 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # calculate loss gradient
                    loss.backward()
                    # TODO 在预热阶段后应用梯度裁剪，防止梯度爆炸
                    # if global_step > warmup_steps:
                    #     torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm = 1.0)
                    # update model weights using loss gradients
                    optimizer.step()
                
                # optional evaluation step
                # ------------------------
                if global_step % eval_freq == 0:
                # TODO if (batch + 1) % 5 == 0:
                    # evaluate model
                    train_loss, valid_loss = self.valid(train_loader, valid_loader, eval_iter)
                    # collect losses and tokens seen
                    train_losses.append(train_loss)
                    valid_losses.append(valid_loss)
                    track_tokens_seen.append(tokens_seen)
                    # logger.info(f"\t\tEpoch {epoch + 1}: Batch {batch + 1} (Step {global_step:06d}): Train loss: {train_loss:.3f}, Val loss {valid_loss:.3f}")
                    
                    # calculate training left time
                    speed = (time.time() - train_start_time) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - batch)
                    logger.info(f'\t\tEpoch {epoch + 1}: Batch {batch + 1} (Step {global_step:06d}): Train loss: {train_loss:.3f}, Val loss {valid_loss:.3f} | Speed: {speed:.4f}s/batch; left time: {left_time:.2f}seconds.')
                    
                    # init epoch iter count
                    iter_count = 0
                    # init train start time
                    train_start_time = time.time()
            """
            # 模型验证
            train_loss = np.average(train_loss)
            vali_loss = self.valid(train_loader, valid_loader, eval_iter)
            logger.info(f"\t\tEpoch: {epoch + 1}, Steps: {train_steps} | Train Loss: {train_loss:.7f}, Vali Loss: {vali_loss:.7f}")
            # 训练/验证损失收集
            train_losses.append(train_loss)
            valid_losses.append(vali_loss)
            """
            
            # 早停机制、模型保存
            early_stopping(epoch = epoch + 1, val_loss = valid_loss, model = self.model, path = model_checkpoint_path)
            if early_stopping.early_stop:
                logger.info(f"\t\tEpoch {epoch + 1}: Early stopping...")
                break
            # 学习率调整
            adjust_learning_rate(optimizer = optimizer, epoch = epoch + 1, args = self.args)
            
            # ------------------------------
            # model inference
            # ------------------------------
            # print a sample text after each epoch
            self.inference(epoch = epoch + 1, setting=setting, load=False)
            
            # calculate one epoch training used time
            logger.info(f"\t\tEpoch {epoch + 1}: cost time: {(time.time() - epoch_start_time):.2f}seconds")
        # ------------------------------
        # 模型训练结果保存、模型加载
        # ------------------------------
        logger.info(f"{43 * '-'}")
        logger.info(f"Training Finished!")
        logger.info(f"{43 * '-'}")
        # calculate all epoch training used time
        logger.info(f"Training Iter {training_iter + 1} cost time: {((time.time() - train_start_time) / 60):.2f}mins")
        
        # plot loss
        logger.info("Plot and save train/valid losses...")
        plot_losses(self.args.train_epochs, track_tokens_seen, train_losses, valid_losses, "loss", results_path)
        
        # model load
        logger.info("Loading best model...")
        self.model.load_state_dict(torch.load(model_checkpoint_path))
        
        # return model and train results
        logger.info("Return training results...")
        return self.model

    def valid(self, train_loader, valid_loader, eval_iter):
        """
        model evaluation
        """
        # logger.info(f"\t\tModel start validating...")
        # inference mode
        self.model.eval()
        # model evaluation
        with torch.no_grad():
            train_loss = calc_loss_loader(
                task_name=self.args.task_name,
                data_loader=train_loader,
                model=self.model, 
                device=self.device,
                num_batches=eval_iter
            )
            valid_loss = calc_loss_loader(
                task_name=self.args.task_name,
                data_loader=valid_loader,
                model=self.model, 
                device=self.device,
                num_batches=eval_iter
            )
        # training mode
        self.model.train()
        
        return train_loss, valid_loss

    def inference(self, epoch, setting: str, load: bool=False, start_context: str="Every effort moves you"):
        """
        model inference
        """
        # logger.info(f"\t\tModel start inference...")
        # 模型加载
        if load:
            logger.info(f"Pretrained model has loaded from: {model_checkpoint_path}")
            model_checkpoint_path = self._get_model_path(setting)
            self.model.load_state_dict(torch.load(model_checkpoint_path)) 
        # inference mode
        self.model.eval()
        # context size
        context_size = self.model.pos_emb.weight.shape[0]
        # start context tokenization
        start_context_encoded = text_to_token_ids(start_context).to(self.device)
        # generate text
        with torch.no_grad():
            completion_id = generate(
                model = self.model, 
                token_idx = start_context_encoded,
                max_new_tokens = self.args.max_new_tokens,
                context_size = context_size,
            )
            completion = token_ids_to_text(completion_id).replace("\n", " ")
            logger.info(f"\t\tEpoch {epoch}: Model inference [start context]: {start_context}, [completion]: {completion}")
        # train mode
        self.model.train()




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
