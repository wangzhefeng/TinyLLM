# -*- coding: utf-8 -*-

# ***************************************************
# * File        : ddp_script.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2025-02-18
# * Version     : 0.1.021821
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
import platform

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from utils.log_util import logger

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


def ddp_setup(rank, world_size):
    """
    function to initialize a distributed process group(1 process/GPU)
    this allows communication among processes

    Args:
        rank (_type_): a unique process ID
        world_size (_type_): total number of processes in the group
    """
    if "MASTER_ADDR" not in os.environ:
        # rank of machine running rank:0 process, assume all GPUs are on the same machine
        os.environ["MASTER_ADDR"] = "localhost"
    if "MASTER_PORT" not in os.environ:
        # any free port on the machine
        os.environ["MASTER_PORT"] = "1234567"
    
    # initialize process group
    if platform.system() == "Windows":
        # disable libuv because PyTorch for Windows isn't built with support
        os.environ["USE_LIBUV"] = "0"
        # gllo: Facebook Collective Communication Library
        init_process_group(backend="gloo", rank=rank, world_size=world_size)
    else:
        # nccl: NVIDIA Collective Communication Library
        init_process_group(backend="nccl", rank=rank, world_size=world_size)
    
    torch.cuda.set_device(rank)




# 测试代码 main 函数
def main(rank, world_size, train_epochs):
    # initialize process groups
    ddp_setup(rank, world_size)
    # data loader
    train_loader, test_loader = None, None
    # model
    model = None
    model.to(rank)
    # optimizer
    optimizer = None
    # wrap model with DDP
    model = DDP(model, device_ids = [rank])
    # the core model is now accessible as model.module
    for epoch in range(train_epochs):
        # set sampler to ensure each epoch has a different shuffle order
        train_loader.sampler.set_epoch(epoch)
        # training mode
        model.train()
        for input_batch, target_batch in train_loader:
            # use rank
            input_batch = input_batch.to(rank)
            target_batch = target_batch.to(rank)
            # forward
            logits = model(input_batch)
            loss = F.cross_entropy(logits, target_batch)
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # log
            logger.info(f"[GPU{rank}] Epoch: {epoch+1:03d}/{train_epochs:03d}"
                        f" | Batchsize {target_batch.shape[0]:03d}"
                        f" | Train/Val Loss: {loss:.2f}")
    model.eval()

    try:
        train_acc = None
        test_acc = None
    except ZeroDivisionError as e:
        raise ZeroDivisionError()

    # cleanly exit distributed mode
    destroy_process_group()

if __name__ == "__main__":
    logger.info(f"torch version: {torch.__version__}")
    logger.info(f"cuda available: {torch.cuda.is_available()}")
    logger.info(f"number of GPUs available: {torch.cuda.device_count()}")

    # spawn new processes: spawn will automatically pass the rank
    torch.manual_seed(123)
    train_epochs = 3
    world_size = torch.cuda.device_count()
    mp.spawn(main, args = (world_size, train_epochs), nprocs=world_size)
