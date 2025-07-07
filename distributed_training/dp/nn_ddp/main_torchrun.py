# -*- coding: utf-8 -*-

# ***************************************************
# * File        : main_torchrun.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-07-07
# * Version     : 1.0.070714
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

import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import destroy_process_group
from torch.distributed.elastic.multiprocessing.errors import record

from distributed_training.utils.ddp_utils import ddp_setup_custom
from distributed_training.ddp.nn_ddp.data_provider import prepare_dataset
from distributed_training.ddp.nn_ddp.model import NeuralNetwork

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]
os.environ['LOG_NAME'] = LOGGING_LABEL
from utils.log_util import logger


def compute_accuracy(model, dataloader, device):
    model = model.eval()
    correct = 0.0
    total_examples = 0

    for idx, (features, labels) in enumerate(dataloader):
        features, labels = features.to(device), labels.to(device)

        with torch.no_grad():
            logits = model(features)
        predictions = torch.argmax(logits, dim=1)
        compare = labels == predictions
        correct += torch.sum(compare)
        total_examples += len(compare)
    return (correct / total_examples).item()


def train(rank, world_size, num_epochs):
    # initialize process group
    ddp_setup_custom(rank, world_size)
    
    # data prepare
    train_loader, test_loader = prepare_dataset()
    # model
    model = NeuralNetwork(num_inputs=2, num_outputs=2)
    model.to(rank)
    # optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
    
    # wrap model with DDP(the core model is now accessible as model.module)
    model = DDP(model, device_ids=[rank], output_device=rank)

    for epoch in range(num_epochs):
        # set sampler to ensure each epoch has a different shuffle order
        train_loader.sampler.set_epoch(epoch)
        # training mode
        model.train()
        for features, labels in train_loader:
            # use rank
            features, labels = features.to(rank), labels.to(rank)
            # forward
            logits = model(features)
            loss = F.cross_entropy(logits, labels)
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # log
            logger.info(f"[GPU{rank}] Epoch: {epoch+1:03d}/{num_epochs:03d}"
                  f" | Batchsize {labels.shape[0]:03d}"
                  f" | Train/Val Loss: {loss:.2f}")

    model.eval()

    try:
        train_acc = compute_accuracy(model, train_loader, device=rank)
        logger.info(f"[GPU{rank}] Training accuracy: {train_acc:.4f}")
        test_acc = compute_accuracy(model, test_loader, device=rank)
        logger.info(f"[GPU{rank}] Test accuracy: {test_acc:.4f}")
    except ZeroDivisionError as e:
        raise ZeroDivisionError(
            f"{e}\n\nThis script is designed for 2 GPUs. You can run it as:\n"
            "torchrun --nproc_per_node=2 DDP-script-torchrun.py\n"
            f"Or, to run it on {torch.cuda.device_count()} GPUs, uncomment the code on lines 103 to 107."
        )
    
    # cleanly exit distributed mode
    destroy_process_group()





# 测试代码 main 函数
@record
def main():
    # Use environment variabels set by torchrun if available, otherwise default to single-process.
    if "WORLD_SIZE" in os.environ:
        world_size = int(os.environ["WORLD_SIZE"])
    else:
        world_size = 1
    
    if "LOCAL_RANK" in os.environ:
        rank = int(os.environ["LOCAL_RANK"])
    elif "RANK" in os.environ:
        rank = int(os.environ["RANK"])
    else:
        rank = 0
    
    # Only print on rank 0 to avoid duplicate prints from each GPU process
    if rank == 0:
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        logger.info(f"Number of GPUs available: {torch.cuda.device_count()}")
    
    # set seed
    torch.manual_seed(123)

    # train epochs
    num_epochs = 3

    # training
    train(rank, world_size, num_epochs)

if __name__ == "__main__":
    main()
