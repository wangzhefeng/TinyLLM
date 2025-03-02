# -*- coding: utf-8 -*-

# ***************************************************
# * File        : pretrain.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2025-02-09
# * Version     : 0.1.020921
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import os
import sys
ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
import time
import math
import warnings
import platform
import argparse

import pandas as pd
import torch
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
from contextlib import nullcontext
from transformers import AutoTokenizer
import wandb

from llm_proj.model import Transformer
from config.Config import ModelConfig
from data_prepare.dataset import PretrainDataset

warnings.filterwarnings("ignore")

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


def Logger(ddp, content):
    if not ddp or dist.get_rank() == 0:
        print(content)


def get_lr(args, it, all):
    warmup_iters = args.warmup_iters
    lr_decay_iters = all
    min_lr = args.learning_rate / 10

    if it < warmup_iters:
        return args.learning_rate * it / warmup_iters
    
    if it > lr_decay_iters:
        return min_lr
    
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))

    return min_lr + coeff * (args.learning_rate - min_lr)


def train_epoch(args, train_loader, epoch, iter_per_epoch, model, optimizer, wandb):
    start_time = time.time()
    for step, (X, Y, loss_mask) in enumerate(train_loader):
        # data move to device
        X = X.to(args.device)
        Y = Y.to(args.device)
        loss_mask = loss_mask.to(args.device)
        # learnint rate
        lr = get_lr(args = args, it = epoch * iter_per_epoch + step, all = args.epochs * iter_per_epoch)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        # ------------------------------
        # 
        # ------------------------------
        with ctx:
            out = model(X, Y)
            loss = out.last_loss / args.accumulation_steps
            loss_mask = loss_mask.view(-1)
            loss = torch.sum(loss * loss_mask) / loss_mask.sum()
        
        scaler.scale(loss).backward()
        # ------------------------------
        # 
        # ------------------------------
        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none = True)

        if step % args.log_interval == 0:
            spend_time = time.time() - start_time
            Logger(
                f"""
                Epoch: [{epoch}/{args.epochs}]({step}/{iter_per_epoch}) 
                loss: {loss.item() * args.accumulation_steps:.3f} 
                lr: {optimizer.param_groups[-1]["lr"]:.7f} 
                epoch_Time: {spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60}min"""
            )
            # wandb setting
            if (wandb is not None) and (not ddp or dist.get_rank() == 0):
                wandb.log({
                    "loss": loss.item() * args.accumulation_steps,
                    "lr": optimizer.param_groups[-1]["lr"],
                    "epoch_Time": spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60
                })
        
        if (step + 1) % args.save_interval == 0 and (not ddp or dist.get_rank() == 0):
            model.eval()
            moe_path = "_moe" if lm_config.use_moe else ""
            ckp = f"{args.save_dir}/pretrain_{lm_config.dim}{moe_path}.pth"

            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
            torch.save(state_dict, ckp)
            model.train()


def init_model(args, lm_config):
    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained("./dataset/tokenizer")
    # model
    model = Transformer(lm_config).to(args.device)
    # moe path
    moe_path = "_moe" if lm_config.use_moe else ""
    # model params counts
    Logger(f"LLM 总参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    return model, tokenizer


def init_distributed_mode():
    if not ddp:
        return
    global ddp_local_rank, DEVICE

    dist.init_process_group(backend = "nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    DEVICE = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(DEVICE)




# 测试代码 main 函数
def main():
    # ------------------------------
    # run
    # ------------------------------
    # torchrun --nproc_per_node 2 pretrain.py

    # ------------------------------
    # user command setting
    # ------------------------------
    parser = argparse.ArgumentParser(description="MiniMind Pretraining")
    parser.add_argument("--out_dir", type=str, default="out", 
                        help="Output directory")
    parser.add_argument("--epochs", type=int, default=20, 
                        help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=64, 
                        help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-4, 
                        help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu",
                        help="Device to use")
    parser.add_argument("--dtype", type=str, default="bfloat16", 
                        help="Data type")
    parser.add_argument("--use_wandb", action="store_true", 
                        help="Use Weights & Biases")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Pretrain", 
                        help="Weights & Biases project name")
    parser.add_argument("--num_workers", type=int, default=1, 
                        help="Number of workers for data loading")
    parser.add_argument("--data_path", type=str, default="./dataset/pretrain/pretrain_data.csv", 
                        help="Path to training data")
    parser.add_argument("--ddp", action="store_true", help="Use DistributedDataParallel")
    parser.add_argument("--accumulation_steps", type=int, default=8, 
                        help="Gradient accumulation steps")
    parser.add_argument("--grad_clip", type=float, default=1.0, 
                        help="Gradient clipping threshold")
    parser.add_argument("--warmup_iters", type=int, default=0, 
                        help="Number of warmup iterations")
    parser.add_argument("--log_interval", type=int, default=100, 
                        help="Logging interval")
    parser.add_argument("--save_interval", type=int, default=1000, 
                        help="Model saving interval")
    parser.add_argument('--local_rank', type=int, default=-1, 
                        help='local rank for distributed training')
    args = parser.parse_args()

    # make sure output directory exists
    os.makedirs(args.out_dir, exist_ok = True)

    # ------------------------------
    # model config
    # ------------------------------
    lm_config = ModelConfig()
    
    # ------------------------------
    # random seed
    # ------------------------------
    torch.manual_seed(1337)

    # ------------------------------
    # ctx
    # ------------------------------
    device_type = "cuda" if "cuda" in args.device else "cpu"
    ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast("cuda")

    # ------------------------------
    # 
    # ------------------------------
    ddp = int(os.environ.get("RANK", -1)) != -1
    ddp_local_rank, DEVICE = 0, "cuda:0"
    if ddp:
        init_distributed_mode()
        args.device = torch.device(DEVICE)
    
    # ------------------------------
    # 
    # ------------------------------
    args.wandb_run_name = f"MiniMind-Pretrain-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"
    if args.use_wandb and (not ddp or ddp_local_rank == 0):
        wandb.init(project = args.wandb_project, name = args.wandb_run_name)
    else:
        wandb = None
    
    # ------------------------------
    # 
    # ------------------------------
    model, tokenizer = init_model()

    # ------------------------------
    # 
    # ------------------------------
    # data
    df = pd.read_csv(args.data_path)
    df = df.sample(frac = 1.0)

    # dataset
    train_ds = PretrainDataset(df, tokenizer, max_length = lm_config.max_seq_len)

    # sampler
    train_sampler = DistributedSampler(train_ds) if ddp else None

    # data loader
    train_loader = DataLoader(
        train_ds,
        batch_size = args.batch_size,
        pin_memory = True,
        drop_last = False,
        shuffle = False,
        num_workers = args.num_workers,
        sampler = train_sampler,
    )
    
    # ------------------------------
    # amp
    # ------------------------------
    scaler = torch.amp.GradScaler("cuda", enabled = (args.dtype in ["float16", "bfloat16"]))

    # ------------------------------
    # optimizer
    # ------------------------------
    optimier = torch.optim.Adam(model.parameters(), lr = args.learning_rate)

    # ------------------------------
    # torch compile model
    # ------------------------------
    if platform.system() != "Windows" and float(torch.__version__.split(".")[0]) >= 2.0:
        Logger("compiling the model... (takes a ~minute)")
        unoptimized_model = model
        model = torch.compile(model)
    
    # ------------------------------
    # DDP
    # ------------------------------
    if ddp:
        model._ddp_params_and_buffers_to_ignore = {"pos_cis"}
        model = DistributedDataParallel(model, device_ids = [ddp_local_rank])

    # ------------------------------
    # model training 
    # ------------------------------
    iter_per_epoch = len(train_loader)
    for epoch in range(args.epochs):
        train_epoch(epoch, wandb)

if __name__ == "__main__":
    main()
