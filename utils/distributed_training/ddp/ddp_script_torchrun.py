# -*- coding: utf-8 -*-

# ***************************************************
# * File        : ddp_script_torchrun.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2025-02-18
# * Version     : 0.1.021823
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
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from utils.log_util import logger

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


def ddp_setup(rank, world_size):
    """
    function to initialize a distributed process group (1 process / GPU)
    this allows communication among processes

    Args:
        rank: a unique process ID
        world_size: total number of processes in the group
    """
    # Only set MASTER_ADDR and MASTER_PORT if not already defined by torchrun
    if "MASTER_ADDR" not in os.environ:
        # rank of machine running rank:0 process, assume all GPUs are on the same machine
        os.environ["MASTER_ADDR"] = "localhost"
    if "MASTER_PORT" not in os.environ:
        # any free port on the machine
        os.environ["MASTER_PORT"] = "1234567"

    # initialize process group
    if platform.system() == "Windows":
        # Disable libuv because PyTorch for Windows isn't built with support
        os.environ["USE_LIBUV"] = "0"
        # gloo: Facebook Collective Communication Library
        init_process_group(backend="gloo", rank=rank, world_size=world_size)
    else:
        # nccl: NVIDIA Collective Communication Library
        init_process_group(backend="nccl", rank=rank, world_size=world_size)

    torch.cuda.set_device(rank)


class ToyDataset(Dataset):

    def __init__(self, X, y):
        self.features = X
        self.labels = y

    def __getitem__(self, index):
        one_x = self.features[index]
        one_y = self.labels[index]
        return one_x, one_y

    def __len__(self):
        return self.labels.shape[0]


class NeuralNetwork(torch.nn.Module):

    def __init__(self, num_inputs, num_outputs):
        super(NeuralNetwork, self).__init__()

        self.layers = torch.nn.Sequential(
            # 1st hidden layer
            torch.nn.Linear(num_inputs, 30),
            torch.nn.ReLU(),

            # 2nd hidden layer
            torch.nn.Linear(30, 20),
            torch.nn.ReLU(),

            # output layer
            torch.nn.Linear(20, num_outputs),
        )

    def forward(self, x):
        logits = self.layers(x)
        return logits


def prepare_dataset():
    X_train = torch.tensor([
        [-1.2, 3.1],
        [-0.9, 2.9],
        [-0.5, 2.6],
        [2.3, -1.1],
        [2.7, -1.5]
    ])
    y_train = torch.tensor([0, 0, 0, 1, 1])

    X_test = torch.tensor([
        [-0.8, 2.8],
        [2.6, -1.6],
    ])
    y_test = torch.tensor([0, 1])

    # Uncomment these lines to increase the dataset size to run this script on up to 8 GPUs:
    # factor = 4
    # X_train = torch.cat([X_train + torch.randn_like(X_train) * 0.1 for _ in range(factor)])
    # y_train = y_train.repeat(factor)
    # X_test = torch.cat([X_test + torch.randn_like(X_test) * 0.1 for _ in range(factor)])
    # y_test = y_test.repeat(factor)

    train_ds = ToyDataset(X_train, y_train)
    test_ds = ToyDataset(X_test, y_test)

    train_loader = DataLoader(
        dataset=train_ds,
        batch_size=2,
        shuffle=False,  # NEW: False because of DistributedSampler below
        pin_memory=True,
        drop_last=True,
        # NEW: chunk batches across GPUs without overlapping samples:
        sampler=DistributedSampler(train_ds)  # NEW
    )
    test_loader = DataLoader(
        dataset=test_ds,
        batch_size=2,
        shuffle=False,
    )
    return train_loader, test_loader


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




# 测试代码 main 函数
def main(rank, world_size, train_epochs):
    # initialize process groups
    ddp_setup(rank, world_size)
    # data loader
    train_loader, test_loader = prepare_dataset()
    # model
    model = NeuralNetwork(num_inputs=2, num_outputs=2)
    model.to(rank)
    # optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
    # wrap model with DDP
    model = DDP(model, device_ids = [rank])
    # the core model is now accessible as model.module
    for epoch in range(train_epochs):
        # Set sampler to ensure each epoch has a different shuffle order
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
            logger.info(f"[GPU{rank}] Epoch: {epoch+1:03d}/{train_epochs:03d}"
                        f" | Batchsize {labels.shape[0]:03d}"
                        f" | Train/Val Loss: {loss:.2f}")
    model.eval()

    try:
        train_acc = compute_accuracy(model, train_loader, device=rank)
        logger.info(f"[GPU{rank}] Training accuracy: {train_acc}")
        test_acc = compute_accuracy(model, test_loader, device=rank)
        logger.info(f"[GPU{rank}] Test accuracy: {test_acc}")
    except ZeroDivisionError as e:
        raise ZeroDivisionError()
    
    # cleanly exit distributed mode
    destroy_process_group()

if __name__ == "__main__":
    # Use environment variables set by torchrun if available, otherwise default to single-process.
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
        logger.info(f"torch version: {torch.__version__}")
        logger.info(f"cuda available: {torch.cuda.is_available()}")
        logger.info(f"number of GPUs available: {torch.cuda.device_count()}")

    torch.manual_seed(123)
    train_epochs = 3
    main(rank, world_size, train_epochs)
