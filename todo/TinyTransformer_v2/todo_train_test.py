# -*- coding: utf-8 -*-

# ***************************************************
# * File        : train_test.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-01-01
# * Version     : 1.0.010118
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# * TODO        : 1.
# ***************************************************

__all__ = []

# python libraries
import os
import sys
ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer import Trnasformer

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


# model params
d_model = 512
heads = 8
N = 6
EN_TEXT = None
ZH_TEXT = None
src_vocab = len(EN_TEXT.vocab)
trg_vocab = len(ZH_TEXT.vocab)

# model
model = Trnasformer(src_vocab, trg_vocab, d_model, N, heads, dropout=0.1)
# params init
for p in model.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)
# optimizer
optim = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
# loss function
criterion = torch.nn.CrossEntropyLoss(ignore_index=0)

# TODO model training


def create_mask():
    pass


def tokenize_en(sentence):
    return [tok.text for tok in EN_TEXT.tokenizer(sentence)]


def train(model, 
          optim, 
          criterion,
          train_loader, 
          target_pad,
          epochs, print_every=100):
    """
    模型训练

    Args:
        epochs (_type_): _description_
        print_every (int, optional): _description_. Defaults to 100.
    """
    # 模型开启训练模式
    model.train()

    start = time.time()
    temp = start
    total_loss = 0
    for epoch in range(epochs):
        for i, batch in enumerate(train_loader):
            src = batch.English.transpose(0, 1)
            trg = batch.Chinese.transpose(0, 1)
            # 将输入的英语句子中的所有单词翻译成中文，除了最后一个单词，因为它正在使用每个单词来预测下一个单词
            trg_input = trg[:, :-1]
            # 试图预测单词
            targets = trg[:, 1:].contiguous().view(-1)
            # 使用掩码代码创建函数来制作掩码
            src_mask, trg_mask = create_mask(src, trg_input)
            # 前向传播
            preds = model(src, trg_input, src_mask, trg_mask)
            optim.zero_grad()
            # 计算损失
            loss = criterion(preds.view(-1, preds.size(-1)), targets, ignore_index=target_pad)
            # 反向传播
            loss.backward()
            # 更新参数
            optim.step()

            total_loss += loss.item()
            if (i + 1) % print_every == 0:
                loss_avg = total_loss / print_every
                print(f"Epoch: {epoch+1}, Step: {i+1}, Loss: {loss_avg}, 
                    Time: {(time.time() - start) // 60}, 
                    {time.time() - temp} per {print_every}")
                total_loss = 0
                temp = time.time()


def translate(model, src, input_pad, max_len=80, custom_string=False):
    """
    模型测试
    """
    # 模型开启测试模式
    model.eval()
    if custom_string:
        src = tokenize_en(src)
        sentence = torch.LongTensor([[EN_TEXT.vocab.stoi[tok] for tok in sentence]])
        src_mask = (src != input_pad).unsqueeze(-2)
        e_outputs = model.encoder(src, src_mask)
        outputs = torch.zeros(max_len).type_as(src.data)
        outputs[0] = torch.LongTensor([ZH_TEXT.vocab.stoi["<sos>"]])

    for i in range(1, max_len):
        trg_mask = np.triu(np.ones(1, i, i), k=1).astype("uint8")
        trg_mask = torch.from_numpy(trg_mask) == 0
        out = model.out(model.decoder(outputs[:i].unsqueeze(0), e_outputs, src_mask, trg_mask))
        out = F.softmax(out, dim=-1)
        val, ix = out[:, -1].data.topk(1)
        outputs[i] = ix[0][0]
        if ix[0][0] == ZH_TEXT.vocab.stoi["<eos>"]:
            break
    return " ".join([ZH_TEXT.vocab.itos[ix] for ix in outputs[:i]])



# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
