# -*- coding: utf-8 -*-

# ***************************************************
# * File        : pretrain.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2024-09-26
# * Version     : 0.1.092602
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
import math
import time
from contextlib import nullcontext
from functools import partial

import torch
from llama_model import Transformer, ModelArgs, Task

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


# ------------------------------
# 配置参数
# ------------------------------
# I/O 配置，用于定义输出目录和训练时的日志记录与评估设置
out_dir = "./tiny-universe/TinyLlama/output"  # 模型输出保存路径
eval_interval = 2000  # 评估间隔步数
log_interval = 1  # 日志记录间隔步数
eval_iters = 100  # 每次评估时迭代的步数
eval_only = False  # 如果为 True，脚本在第一次评估后立即退出
always_save_checkpoint = False  # 如果为True，在每次评估后总是保存检查点
init_from = "scratch"  # 可以选择从头开始训练（'scratch'）或从已有的检查点恢复（'resume'）

# 数据配置
batch_size = 128  # 每个微批次的样本数量，如果使用梯度累积，实际批次大小将更大
max_seq_len = 256  # 最大序列长度
vocab_size = 4096  # 自定义词汇表大小

# 模型配置
dim = 288  # 模型的隐藏层维度
n_layers = 8  # Transformer 的层数
n_heads = 8  # 注意力头的数量
n_group = 4  # 模型分组
multiple_of = 32  # 在某些层的维度必须是该数的倍数
dropout = 0.0  # Dropout 概率

# AdamW 优化器配置
gradient_accumulation_steps = 4  # 梯度累积步数，用于模拟更大的批次
learning_rate = 5e-4  # 最大学习率
max_iters = 100000  # 总的训练迭代次数
weight_decay = 1e-1  # 权重衰减系数
beta1 = 0.9  # AdamW 优化器的 β1 参数
beta2 = 0.95  # AdamW 优化器的 β2 参数
grad_clip = 1.0  # 梯度裁剪阈值，0表示不裁剪
# 学习率衰减配置
decay_lr = True  # 是否启用学习率衰减
warmup_iters = 1000  # 学习率预热的步数

# 系统设置
device = "cuda:0"  # 设备选择：'cpu'，'cuda'，'cuda:0' 等
dtype = "bfloat16"  # 数据类型：'float32'，'bfloat16'，'float16'

# 获取配置参数的键值对，便于后续的日志记录
config_keys = [
    k 
    for k, v in globals().items()
    if not k.startswith("_") and isinstance(v, (int, float, bool, str))
]
# 保存配置到字典中，便于日志记录
config = {k: globals()[k] for k in config_keys}


# ------------------------------
#
# ------------------------------
# 固定一些超参数的默认值
lr_decay_iters = max_iters  # 学习率衰减步数，设置为等于最大迭代步数
min_lr = 0.0  # 最小学习率，建议为学习率的十分之一
vocab_source = 'custom'  # 词汇表来源
master_process = True  # 用于区分主进程
seed_offset = 0  # 随机种子偏移量
ddp_world_size = 1  # 分布式数据并行的世界大小
tokens_per_iter = batch_size * max_seq_len  # 每次迭代处理的 token 数

# 设置随机种子，确保可重复性
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True  # 允许在 matmul 上使用 tf32
torch.backends.cudnn.allow_tf32 = True  # 允许在 cudnn 上使用 tf32
device_type = "cuda" if "cuda" in device else "cpu"  # 用于自动选择设备类型
ptdtype = torch.float16  # 设置训练时使用的数据类型


# ------------------------------
# 
# ------------------------------
# 混合精度训练相关
ctx = (
    nullcontext()
    if device_type == "cpu"
    else torch.amp.autocast(device_type = device_type, dtype = ptdtype)
)


# task-specific setup
iter_batches = partial(
    Task.iter_batches,
    batch_size = batch_size,
    max_seq_len = max_seq_len,
    vocab_size = vocab_size,
    vocab_source = vocab_source,
    device = device,
    num_workers = 0,
)


# 模型参数初始化
model_args = dict(
    dim = dim,
    n_layers = n_layers,
    n_heads = n_heads,
    n_group = n_group,
    vocab_size = vocab_size,
    multiple_of = multiple_of,
    max_seq_len = max_seq_len,
    dropout = dropout,
)
gptconf = ModelArgs(**model_args)


# 模型初始化
model = Transformer(gptconf)
model.to(device)


# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.amp.GradScaler(device="cuda", enabled = (dtype == "float16"))


# optimizer
optimizer = model.configure_optimizers(
    weight_decay, 
    learning_rate, 
    (beta1, beta2), 
    device_type
)


# 定义 eval 流程
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        batch_iter = iter_batches(split = split)
        losses = torch.zeros(eval_iters)  # keep on CPU
        for k in range(eval_iters):
            X, Y = next(batch_iter)
            with ctx:
                logits = model(X, Y)
                loss = raw_model.last_loss
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    
    return out


# 定义学习率
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    
    return min_lr + coeff * (learning_rate - min_lr)




# ------------------------------
# training loop
# ------------------------------
# training data
train_batch_iter = iter_batches(split = "train")
X, Y = next(train_batch_iter)  # fetch the very first batch

# traing params
iter_num = 0
best_val_loss = 1e9
t0 = time.time()
local_iter_num = 0  # number of iterations in the lifetime of this process
raw_model = model  # unwrap DDP container if needed
running_mfu = -1.0

# train loop
while True:
    # 获取当前 step 的学习率
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    
    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if losses["val"] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses["val"]
            if iter_num > 0:
                checkpoint = {
                    "model": raw_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "model_args": model_args,
                    "iter_num": iter_num,
                    "best_val_loss": best_val_loss,
                    "config": config,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, Path(out_dir).joinpath("ckpt.pt"))
    
    if iter_num == 0 and eval_only:
        break
   
    # 前向更新过程，使用了梯度累积(检查点)
    for micro_step in range(gradient_accumulation_steps):
        with ctx:  # 混合精度相关
            logits = model(X, Y)
            loss = raw_model.last_loss
            loss = loss / gradient_accumulation_steps
        X, Y = next(train_batch_iter)
        # 反向传播
        scaler.scale(loss).backward()
    
    # 梯度阶段
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)
    
    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        # get loss as float, scale up due to the divide above. note: this is a CPU-GPU sync point
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5:  # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
        print(f"{iter_num} | loss {lossf:.4f} | lr {lr:e} | {dt*1000:.2f}ms | mfu {running_mfu*100:.2f}%")   # mfu表示模型浮点运算利用率
    iter_num += 1
    local_iter_num += 1
    
    # termination conditions
    if iter_num > max_iters:
        break




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
