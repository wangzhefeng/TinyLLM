# -*- coding: utf-8 -*-

# ***************************************************
# * File        : calc_accuracy.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2025-02-23
# * Version     : 0.1.022300
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

import torch

from utils.log_util import logger

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


def calc_accuracy_loader(dataloader, model, device, num_batches = None):
    # model eval
    model.eval()
    # correct predictions and number of examples
    correct_preds, num_examples = 0, 0
    # number of batches
    if num_batches is None:
        num_batches = len(dataloader)
    else:
        num_batches = min(num_batches, len(dataloader))
    # calculate accuracy
    for i, (input_batch, target_batch) in enumerate(dataloader):
        if i < num_batches:
            # data to device
            input_batch, target_batch = input_batch.to(device), target_batch.to(device)
            # model inference
            with torch.no_grad():
                logits = model(input_batch)[:, -1, :]
            # pred labels
            pred_labels = torch.argmax(logits, dim=-1)
            # collect info
            num_examples += pred_labels.shape[0]
            correct_preds += (pred_labels == target_batch).sum().item()
        else:
            break
    
    return correct_preds / num_examples


def calc_accuracy(train_loader, valid_loader, test_loader, model, device):
    train_accuracy = calc_accuracy_loader(train_loader, model, device, num_batches=10)
    valid_accuracy = calc_accuracy_loader(valid_loader, model, device, num_batches=10)
    test_accuracy = calc_accuracy_loader(test_loader, model, device, num_batches=10)

    logger.info(f"Train accuracy: {train_accuracy * 100:.2f}%")
    logger.info(f"Valid accuracy: {valid_accuracy * 100:.2f}%")
    logger.info(f"Test accuracy: {test_accuracy * 100:.2f}%")




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
