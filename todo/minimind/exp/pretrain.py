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
ROOT = str(os.getcwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)
import time
import math
import warnings
import platform
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
from contextlib import nullcontext
from transformers import AutoTokenizer

from exp.exp_basic import Exp_Basic
from minimind.data_provider.data_factory import data_provider
from minimind.model.model_config import ModelConfig
from utils.log_util import logger

warnings.filterwarnings("ignore")

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


class Model(Exp_Basic):

    def __init__(self, args):
        logger.info(f"{40 * '-'}")
        logger.info("Initializing Experiment...")
        logger.info(f"{40 * '-'}")
        super(Model, self).__init__(args)
        
        # tokenizer
        self.tokenizer = self._build_tokenizer()

    def _build_tokenizer(self):
        # tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.args.tokenizer_path)

        return tokenizer

    def _init_distributed_mode(self, base_seed = 1337):
        # is this a ddp run
        ddp = int(os.environ.get("RANK", -1)) != -1
        # default ddp_local_rank and device
        ddp_local_rank, DEVICE = 0, "cuda:0"
        # 设置随机种子
        torch.manual_seed(base_seed)
        torch.cuda.manual_seed(base_seed)
        # ddp 训练设置
        if ddp:
            # init process group
            dist.init_process_group(backend = "nccl")
            # ddp params
            ddp_rank = int(os.environ["RANK"])
            ddp_local_rank = int(os.environ["LOCAL_RANK"])
            ddp_world_size = int(os.environ["WORLD_SIZE"])
            # ddp mode device
            DEVICE = f"cuda:{ddp_local_rank}"
            torch.cuda.set_device(DEVICE)
            # update device setup
            self.device = torch.device(DEVICE)
            # torch.distributed rank
            rank = dist.get_rank()
            # 设置 torch 随机种子
            torch.manual_seed(base_seed + rank)
            # 同时设置 CUDA 的随机种子
            torch.cuda.manual_seed(base_seed + rank)

        return ddp, ddp_local_rank

    def _wandb_setup(self, ddp, ddp_local_rank, wandb_run_name):
        """
        wandb 设置
        """
        if self.args.use_wandb and (not ddp or ddp_local_rank == 0):
            import wandb
            wandb.init(project = self.args.wandb_project, name = wandb_run_name)
        else:
            wandb = None
        
        return wandb
    
    def _build_model(self, model_config, ddp, ddp_local_rank):
        """
        模型构建
        """
        # 模型初始化
        logger.info(f"Initializing model {self.args.model}...")
        model = self.model_dict[self.args.model_name].Model(model_config)
        
        # TODO 多 GPU 训练
        # if self.args.use_gpu and self.args.use_multi_gpu:
        #     model = nn.DataParallel(model, device_ids = self.args.devices)
        
        # TODO ddp 训练
        if ddp:
            model._ddp_params_and_buffers_to_ignore = {"pos_cis"}
            model = DistributedDataParallel(model, device_ids = [ddp_local_rank])
        
        # torch compile model
        if platform.system() != "Windows" and float(torch.__version__.split(".")[0]) >= 2.0:
            logger.info("Compiling the model(takes a minute)...")
            unoptimized_model = model
            model = torch.compile(model)

        # 打印模型参数量
        total = sum([param.nelement() for param in model.parameters()])
        logger.info(f'Number of model parameters: {(total / 1e6):.3f}M')
        
        return model
    
    def _get_data(self, ddp):
        """
        数据集构建
        """
        data_set, data_loader = data_provider(self.args, self.tokenizer, ddp)
        
        return data_set, data_loader
    
    def _select_criterion(self):
        """
        评价指标
        """
        criterion = nn.CrossEntropyLoss(reduction="none")

        return criterion

    def _select_optimizer(self):
        """
        优化器
        """
        optimizer = torch.optim.AdamW(self.model.parameters(), lr = self.args.learning_rate)
        
        return optimizer
    
    def _build_grad_scaler(self):
        """
        Gradient Scaler
        """
        scaler = torch.amp.GradScaler("cuda", enabled=(self.args.dtype in ["float16", "bfloat16"]))

        return scaler
    
    def _build_amp(self):
        """
        混合精度
        """
        ctx = nullcontext() if self.args.gpu_type == "cpu" else torch.amp.autocast("cuda")

        return ctx
    
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
    
    def Logger(self, ddp, content):
        if not ddp or dist.get_rank() == 0:
            logger.info(content)

    # TODO
    # def get_lr(self, it, all):
    #     warmup_iters = self.args.warmup_iters
    #     lr_decay_iters = all
    #     min_lr = self.args.learning_rate / 10

    #     if it < warmup_iters:
    #         return self.args.learning_rate * it / warmup_iters
        
    #     if it > lr_decay_iters:
    #         return min_lr
        
    #     decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    #     assert 0 <= decay_ratio <= 1
    #     coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))

    #     return min_lr + coeff * (self.args.learning_rate - min_lr)

    def get_lr(self, current_step, total_steps, lr):
        return lr / 10 + 0.5 * lr * (1 + math.cos(math.pi * current_step / total_steps)) 

    def train(self, wandb_run_name):
        # 混合精度
        ctx = self._build_amp()
        
        # ddp setup
        ddp, ddp_local_rank = self._init_distributed_mode(base_seed = 1337)
        
        # wandb setup
        wandb = self._wandb_setup(ddp, ddp_local_rank, wandb_run_name)

        # dataset and dataloader
        train_data, train_loader = self._get_data(self.tokenizer, ddp)

        # TODO model config
        model_config = ModelConfig()

        # model
        model = self._build_model(ddp, model_config, ddp_local_rank)
        
        # scaler
        scaler = self._build_grad_scaler()
        
        # loss
        criterion = self._select_criterion()
        
        # optimizer
        optimizer = self._select_optimizer()

        # TODO
        tokens_per_iter = self.args.batch_size * self.args.max_seq_len 
 
        # model training 
        iter_per_epoch = len(train_loader)
        for epoch in range(self.args.train_epochs):
            train_start_time = time.time()
            for step, (X, Y, loss_mask) in enumerate(train_loader):
                # data move to device
                X = X.to(self.device)
                Y = Y.to(self.device)
                loss_mask = loss_mask.to(self.device)
                
                # learnint rate
                lr = self.get_lr(
                    current_step = epoch * iter_per_epoch + step, 
                    total_steps = self.args.train_epoch * iter_per_epoch,
                    lr = self.args.learning_rate,
                )
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr
                
                # TODO
                with ctx:
                    res = model(X)
                    loss = criterion(res.logits.view(-1, res.logits.size(-1)), Y.view(-1))
                    loss = torch.sum(loss * loss_mask) / loss_mask.sum()
                    loss += res.aux_loss
                    loss = loss / self.args.accumulation_steps
                
                # TODO
                scaler.scale(loss).backward()
                if (step + 1) % self.args.accumulation_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.grad_clip)

                    scaler.step(optimizer)
                    scaler.update()

                    optimizer.zero_grad(set_to_none = True)

                # 打印模型日志
                if step % self.args.log_interval == 0:
                    train_spend_time = time.time() - train_start_time
                    self.Logger(
                        'Epoch:[{}/{}]({}/{}) loss:{:.3f} lr:{:.12f} epoch_Time:{}min:'.format(
                            epoch + 1,
                            self.args.train_epochs,
                            step,
                            iter_per_epoch,
                            loss.item() * self.args.accumulation_steps,
                            optimizer.param_groups[-1]['lr'],
                            train_spend_time / (step + 1) * iter_per_epoch // 60 - train_spend_time // 60
                        )
                    )
                    
                    # wandb setting
                    if (wandb is not None) and (not ddp or dist.get_rank() == 0):
                        wandb.log({
                            "loss": loss.item() * self.args.accumulation_steps,
                            "lr": optimizer.param_groups[-1]["lr"],
                            "epoch_Time": train_spend_time / (step + 1) * iter_per_epoch // 60 - train_spend_time // 60
                        })
                
                # 模型保存
                if (step + 1) % self.args.save_interval == 0 and (not ddp or dist.get_rank() == 0):
                    model.eval()
                    moe_path = "_moe" if model_config.use_moe else ""
                    ckp = f"{self.args.checkpoints}/pretrain_{model_config.hidden_size}{moe_path}.pth"

                    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                        state_dict = model.module.state_dict()
                    else:
                        state_dict = model.state_dict()
                    # 半精度保存
                    state_dict = {k: v.half() for k, v in state_dict.items()}
                    torch.save(state_dict, ckp)
                    model.train()




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
