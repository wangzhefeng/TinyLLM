# -*- coding: utf-8 -*-

# ***************************************************
# * File        : train.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2025-02-20
# * Version     : 0.1.022022
# * Description : description
# * Link        : https://github.com/rasbt/LLMs-from-scratch/blob/main/appendix-D/01_main-chapter-code/appendix-D.ipynb
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import os
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)
import math


import torch

from utils.log_util import logger

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


# ------------------------------
# train process
# ------------------------------
def train(model, 
          optimizer, 
          train_loader, 
          valid_loader, 
          device, 
          train_epochs, 
          eval_freq, 
          eval_iter, 
          start_context, 
          tokenizer, 
          warmup_steps, 
          initial_lr = 3e-5, 
          min_lr = 1e-6):
    # Initialize to track training result
    train_losses, val_losses = [], [] 
    track_tokens_seen = []
    tokens_seen = 0
    global_step = -1

    # 从优化器中获取最大学习率
    peak_lr = optimizer.param_groups[0]["lr"]
    # 计算训练过程中总的迭代次数
    total_training_steps = len(train_loader) * train_epochs
    # 计算warmup阶段的迭代次数
    lr_increment = (peak_lr - initial_lr) / warmup_steps
    track_lrs = []
    for epoch in range(train_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            global_step += 1

            # 根据当前阶段（预热或余弦衰减）调整学习率
            if global_step < warmup_steps:
                # 线性预热
                lr = initial_lr + global_step * lr_increment
            else:
                # 预热后余弦衰减
                progress = (global_step - warmup_steps) / (total_training_steps - warmup_steps)
                lr = min_lr + (peak_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
            # 将计算出的学习率应用到优化器中
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
            # 记录当前学习率
            track_lrs.append(lr)

            # 计算loss，更新权重
            loss = _calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()

            # 在预热阶段后应用梯度裁剪，防止梯度爆炸
            BOOK_VERSION = True
            if BOOK_VERSION:
                if global_step > warmup_steps:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 1.0)
            else:
                if global_step >= warmup_steps:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 1.0)
            
            # 更新权重
            optimizer.step()
            # 记录 tokens_seen
            tokens_seen += input_batch.numel()
            # 定期对训练和验证集进行评估
            if global_step % eval_freq == 0:
                train_loss, val_loss = _evaluate_model(model, train_loader, valid_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                logger.info(f"Epoch {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")
        # 生成并打印模型的采样输出以监控训练进展
        _generate_and_print_samples(model, tokenizer, start_context, device, num_samples = 10)

    return train_losses, val_losses, track_tokens_seen, track_lrs




# 测试代码 main 函数
def main():
    peak_lr = 0.001  # 书中原始设置为 5e-4，这是一个错误
    optimizer = torch.optim.AdamW(model.parameters(), lr=peak_lr, weight_decay=0.1)

    train_losses, val_losses, tokens_seen, lrs = train(
        model, train_loader, val_loader, optimizer, device, n_epochs=n_epochs,
        eval_freq=5, eval_iter=1, start_context="Every effort moves you",
        tokenizer=tokenizer, warmup_steps=warmup_steps, 
        initial_lr=1e-5, min_lr=1e-5
    )

if __name__ == "__main__":
    main()
