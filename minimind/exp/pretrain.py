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

import pandas as pd
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
from contextlib import nullcontext
from transformers import AutoTokenizer
import wandb

from exp.exp_basic import Exp_Basic
from minimind.data_provider.data_factory import data_provider
from minimind.model.Config import ModelConfig
from minimind.model.model import ModelForCausalLM
from utils.log_util import logger

warnings.filterwarnings("ignore")

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


class Model(Exp_Basic):

    def __init__(self, args):
        logger.info(f"{40 * '-'}")
        logger.info("Initializing Experiment...")
        logger.info(f"{40 * '-'}")
        super(Model, self).__init__(args)

    def _build_tokenizer(self):
        # tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.args.tokenizer_path)

        return tokenizer

    def _ddp(self):
        # is this a ddp run?
        ddp = int(os.environ.get("RANK", -1)) != -1
        ddp_local_rank, device = 0, "cuda:0"

        return ddp

    def _build_model(self):
        """
        模型构建
        """
        # 模型初始化
        logger.info(f"Initializing model {self.args.model}...")
        model = self.model_dict[self.args.model].Model(self.args)
        
        
        # 多 GPU 训练
        if self.args.use_gpu and self.args.use_multi_gpu:
            model = nn.DataParallel(model, device_ids = self.args.devices)
        
        # TODO
        if self.ddp:
            model._ddp_params_and_buffers_to_ignore = {"pos_cis"}
            model = DistributedDataParallel(model, device_ids = [ddp_local_rank])
        
        # 打印模型参数量
        total = sum([param.nelement() for param in model.parameters()])
        logger.info(f'Number of model parameters: {(total / 1e6):.2f}M')
        
        return model
    
    def _get_data(self, tokenizer, ddp):
        """
        数据集构建
        """
        data_set, data_loader = data_provider(self.args, tokenizer, ddp)
        
        return data_set, data_loader
    
    def _select_criterion(self):
        """
        评价指标
        """
        pass

    def _select_optimizer(self):
        """
        优化器
        """
        optimizer = torch.optim.AdamW(self.model.parameters(), lr = self.args.learning_rate)
        
        return optimizer
    
    def _build_scaler(self):
        scaler = torch.GradScaler("cuda", enabled=(self.args.dtype in ["float16", "bfloat16"]))

        return scaler
    
    def _get_model_path(self, setting):
        """
        模型保存路径
        """
        # 模型保存路径
        model_path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(model_path, exist_ok=True)
        # 最优模型保存路径
        model_checkpoint_path = os.path.join(model_path, "checkpoint.pth")
        
        return model_checkpoint_path
    
    def _get_test_results_path(self, setting):
        """
        结果保存路径
        """
        results_path = os.path.join(self.args.test_results, setting)
        os.makedirs(results_path, exist_ok=True)
        
        return results_path
    
    def _model_forward(self):
        return None
    
    def Logger(self, ddp, content):
        if not ddp or dist.get_rank() == 0:
            logger.info(content)

    # TODO
    def get_lr(self, it, all):
        warmup_iters = self.args.warmup_iters
        lr_decay_iters = all
        min_lr = self.args.learning_rate / 10

        if it < warmup_iters:
            return self.args.learning_rate * it / warmup_iters
        
        if it > lr_decay_iters:
            return min_lr
        
        decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))

        return min_lr + coeff * (self.args.learning_rate - min_lr)

    def get_lr(self, current_step, total_steps, lr):
        return lr / 10 + 0.5 * lr * (1 + math.cos(math.pi * current_step / total_steps))

    def train_epoch(self, epoch, wandb):
        start_time = time.time()
        
        ctx = nullcontext() if self.args.device_type == "cpu" else torch.amp.autocast("cuda")
        for step, (X, Y, loss_mask) in enumerate(train_loader):
            # data move to device
            X = X.to(self.device)
            Y = Y.to(self.device)
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

    def train(self):
        # ------------------------------
        # TODO model config
        # ------------------------------
        lm_config = ModelConfig()
        
        tokens_per_iter = self.args.batch_size * self.args.max_seq_len
        # ------------------------------
        # random seed
        # ------------------------------
        # ------------------------------
        # 
        # ------------------------------
        ddp = int(os.environ.get("RANK", -1)) != -1
        ddp_local_rank, DEVICE = 0, "cuda:0"
        if ddp:
            self.init_distributed_mode()
            self.device = torch.device(DEVICE)
        # ------------------------------
        # 
        # ------------------------------
        if self.args.use_wandb and (not ddp or ddp_local_rank == 0):
            wandb.init(project = self.args.wandb_project, name = self.args.wandb_run_name)
        else:
            wandb = None 
        # ------------------------------
        # amp
        # ------------------------------
        scaler = torch.amp.GradScaler("cuda", enabled = (args.dtype in ["float16", "bfloat16"]))
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
        # model training 
        iter_per_epoch = len(train_loader)
        for epoch in range(self.args.train_epochs):
            self.train_epoch(epoch, wandb)

    def init_model(self, lm_config): 
        # model
        model = ModelForCausalLM(lm_config).to(self.device)
        # model params counts
        self.Logger(f"模型可训练总参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} 百万")
        
        return model

    def init_distributed_mode(self, ddp):
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
    pass

if __name__ == "__main__":
    main()
