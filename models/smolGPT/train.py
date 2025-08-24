# -*- coding: utf-8 -*-

# ***************************************************
# * File        : train.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-01-31
# * Version     : 1.0.013122
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
import time
import math
from functools import partial


import torch
from torch.utils.tensorboard.writer import SummaryWriter

from config import ModelConfig, TrainingConfig
from dataset import Task
from model import GPT
from utils.log_util import logger

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


# setting & config
train_config = TrainingConfig()
out_dir = "out/"
writer = SummaryWriter(log_dir=Path(out_dir).joinpath("logs"))
resume = False

# tokens per iter
tokens_per_iter = (
    train_config.gradient_accumulation_steps
    * train_config.batch_size
    * ModelConfig.block_size
)
logger.info("Tokens per iteration: ", tokens_per_iter)

# random seed & precision setting
torch.manual_seed(42)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")


ctx = torch.autocast(train_config.device, dtype=torch.bfloat16)

# model args
model_args = dict(
    n_layer = ModelConfig.n_layer,
    n_head = ModelConfig.n_head,
    embed_dim = ModelConfig.embed_dim,
    block_size = ModelConfig.block_size,
    bias = ModelConfig.bias,
    vocab_size = ModelConfig.vocab_size,
    dropout = ModelConfig.dropout,
)

# iter batches
iter_batches = partial(
    Task.iter_batches,
    batch_size=train_config.batch_size,
    max_seq_len=ModelConfig.block_size,
    device=train_config.device,
    num_workers=0,
)

# resume mechanism
best_val_loss = 1e9
if resume:
    ckpt_path = Path(out_dir).joinpath("ckpt.pt")
    checkpoint = torch.load(ckpt_path, map_location=train_config.device)
    gptconf = ModelConfig(**checkpoint["model_args"])
    model = GPT(gptconf)
    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    best_val_loss = checkpoint["best_val_loss"]

    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    logger.info("Loaded checkpoint")
else:
    gptconf = ModelConfig(**model_args)
    model = GPT(gptconf).to(train_config.device)

scaler = torch.GradScaler(enabled=(train_config.dtype == "float16"))
optimizer = model.configure_optimizers(
    train_config.weight_decay,
    train_config.learning_rate,
    (train_config.beta1, train_config.beta2),
    train_config.device,
)

if train_config.compile:
    model = torch.compile(model)


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(train_config.eval_iters)
        batch_iter = iter_batches(split=split)
        for k in range(train_config.eval_iters):
            X, Y = next(batch_iter)
            with ctx:
                _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


def get_lr(it):
    if it < train_config.warmup_iters:
        return train_config.learning_rate * (it + 1) / (train_config.warmup_iters + 1)
    if it > train_config.lr_decay_iters:
        return train_config.min_lr
    decay_ratio = (it - train_config.warmup_iters) / (
        train_config.lr_decay_iters - train_config.warmup_iters
    )
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return train_config.min_lr + coeff * (
        train_config.learning_rate - train_config.min_lr
    )


params = sum([p.numel() for p in model.parameters() if p.requires_grad])
logger.info("Number of params:", params)

train_batch_iter = iter_batches(split="train")
X, Y = next(train_batch_iter)

iter_num = 0
t0 = time.time()
while True:
    lr = get_lr(iter_num) if train_config.decay_lr else train_config.learning_rate
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    if iter_num % train_config.eval_interval == 0:
        losses = estimate_loss()
        logger.info(
            f"step {iter_num}: train_loss {losses['train']:.4f}, val_loss {losses['val']:.4f}"
        )
        writer.add_scalar("train_loss", losses["train"], iter_num)
        writer.add_scalar("val_loss", losses["val"], iter_num)
        writer.add_scalar("lr", lr, iter_num)

        if losses["val"] < best_val_loss:
            best_val_loss = losses["val"]
            checkpoint = {
                "model": model.state_dict(),
                "model_args": model_args,
                "iter_num": iter_num,
                "best_val_loss": best_val_loss,
            }
            logger.info(f"Saving checkpoint to {out_dir}")
            torch.save(checkpoint, Path(out_dir).joinpath("ckpt.pt"))

    for micro_step in range(train_config.gradient_accumulation_steps):
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / train_config.gradient_accumulation_steps
        X, Y = next(train_batch_iter)
        scaler.scale(loss).backward()

    if train_config.grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    t1 = time.time()
    dt = t1 - t0
    t0 = t1

    if iter_num % train_config.log_interval == 0:
        lossf = loss.item() * train_config.gradient_accumulation_steps
        logger.info(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms")

    iter_num += 1

    if iter_num > train_config.max_iters:
        break

writer.close()




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
