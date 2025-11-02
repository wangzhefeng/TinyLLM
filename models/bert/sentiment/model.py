# -*- coding: utf-8 -*-

# ***************************************************
# * File        : model.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-10-18
# * Version     : 1.0.101821
# * Description : description
# * Link        : https://github.com/datawhalechina/hand-bert/blob/main/others/Sentiment/model.py
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import os
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
# from transformers import BertModel

from utils.log_util import logger

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


class BertClassifier(nn.Module):
    def __init__(self, model_path):
        super().__init__()
        D_in, H, D_out = 768, 100, 2
        self.bert = BertModel.from_pretrained(model_path)
        self.classifier = nn.Sequential(
            nn.Linear(D_in, H),
            nn.ReLU(),
            nn.Linear(H, D_out),
        )

    def forward(self, input_ids, attention_mask):
        # BERT model
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # 为分类任务提取标记 [CLS] 的最后隐藏状态，因为要连接传到全连接层
        last_hidden_state_cls = outputs[0][:, 0, :]
        logits = self.classifier(last_hidden_state_cls)

        return logits

def initialize_model(device, train_dataloader, bert_path, epochs=10):
    # model
    bert_classifier = BertClassifier(bert_path)
    bert_classifier.to(device)
    # optimizer
    optimizer = AdamW(bert_classifier.parameters(), lr=5e-5, eps=1e-8)
    # train steps
    total_steps = len(train_dataloader) * epochs
    # learning rate scheduler
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    
    return bert_classifier, optimizer, scheduler

def train(model, device, optimizer, scheduler, train_dataloader, test_dataloader=None, epochs=10, evaluation=False):
    for epoch_i in range(epochs):
        print(f"{'Epoch':^7} | {'per 10 epoch Batch':^9} | {'train Loss':^12} | {'test Loss':^10} | {'train acc':^9} | {'time':^9}")

def evaluate(model, test_dataloader, loss_fn, device):
    pass




# 测试代码 main 函数
def main():
    print(f"{'Epoch':^7} | {'per 10 epoch Batch':^9} | {'train Loss':^12} | {'test Loss':^10} | {'train acc':^9} | {'time':^9}")

if __name__ == "__main__":
    main()
