# -*- coding: utf-8 -*-

# ***************************************************
# * File        : transformer.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-01-01
# * Version     : 1.0.010116
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

__all__ = []

# python libraries
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


class FeedForward(nn.Module):

    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        super().__init__()

        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear1(x)))
        x = self.linear2(x)
        return x


class LayerNorm(nn.Module):

    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        
        self.size = d_model
        # 层归一化包含两个可以学习的参数
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.beta = nn.Parameter(torch.zeros(self.size))
        self.eps = eps

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
            / (x.std(dim=-1, keepdim=True) + self.eps) + self.beta
        
        return norm


class Embedder(nn.Module):
    """
    词嵌入层，将token索引转换为向量表示
    
    Args:
        vocab_size (int): 词汇表大小
        d_model (int): 嵌入维度
    """
    
    def __init__(self, vocab_size, d_model):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model
    
    def forward(self, x):
        """
        前向传播
        Args:
            x (torch.Tensor): 输入token索引，shape: (batch_size, seq_len)
        
        Returns:
            torch.Tensor: 词嵌入向量，shape: (batch_size, seq_len, d_model)
        """
        return self.embedding(x) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))


class PositionalEncoder(nn.Module):

    def __init__(self, d_model = 512, max_seq_len = 80):
        """
        Args:
            d_model (_type_): 词嵌入维度
            max_seq_len (int, optional): 最大序列长度. Defaults to 80.
        """
        super().__init__()
        
        self.d_model = d_model
        # 根据 pos 和 i 创建一个常量 PE 矩阵
        pe = torch.zeros(max_seq_len, d_model)  # size: (max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** (i / d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** (i / d_model)))
        pe = pe.unsqueeze(0)  # size: (1, max_seq_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # 使得单词嵌入表示相对大一些
        x = x * math.sqrt(self.d_model)
        # 增加位置常量到单词嵌入表示中
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :].require_grad_(False)
        return x


class MultiHeadAttention(nn.Module):
    
    def __init__(self, d_model, heads, dropout = 0.1):
        super().__init__()
        # params
        self.d_model = d_model
        self.h = heads
        self.d_k = d_model // heads
        # layers
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    def attention(self, q, k, v, d_k, mask=None, dropout=None):
        # QK/sqrt(d_k)
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
        # 掩盖掉那些为了填补长度增加的单元，使其通过 softmax 计算后为 0
        if mask is not None:
            mask = mask.unsequeeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        # softmax(QK/sqrt(d_k))
        scores = F.softmax(scores, dim=-1)
        # dropout
        if dropout is not None:
            scores = dropout(scores)
        # softmax(QK/sqrt(d_k))V
        output = torch.matmul(scores, v)

        return output

    def forward(self, q, k, v, mask=None):
        # batch size
        bs = q.size(0)
        # 进行线性操作划分成 h 个头, 并进行矩阵转置
        # q: (bs, nq, d_model) -> (bs, nq, h, d_k) -> (bs, h, nq, d_k)
        # k: (bs, nk, d_model) -> (bs, nk, h, d_k) -> (bs, h, nk, d_k)
        # v: (bs, nk, d_model) -> (bs, nk, h, d_k) -> (bs, h, nk, d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k).transpose(1, 2)
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k).transpose(1, 2)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k).transpose(1, 2)
        # 计算 attention
        scores = self.attention(q, k, v, self.d_k, mask, self.dropout)
        # 链接多个头，并输出到最后的线性层
        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        output = self.out(concat)

        return output


class DecoderLayer(nn.Module):
    
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.attn1 = MultiHeadAttention(d_model, heads, dropout=dropout)
        self.attn2 = MultiHeadAttention(d_model, heads, dropout=dropout)
        self.ff = FeedForward(d_model, d_ff=2048, dropout=dropout)
    
    def forward(self, x, e_outputs, src_mask, trg_mask):
        # masked multi-head attention
        attn_output_1 = self.attn1(x, x, x, trg_mask)
        attn_output_1 = self.dropout1(attn_output_1)
        # add & norm
        x = x + attn_output_1
        x = self.norm1(x)

        # multi-head attention
        attn_output_2 = self.attn2(x, e_outputs, e_outputs, src_mask)
        attn_output_2 = self.dropout2(attn_output_2)
        # add & norm
        x = x + attn_output_2
        x = self.norm2(x)

        # feed forward
        ff_output = self.ff(x)
        ff_output = self.dropout3(ff_output)
        # add & norm
        x = x + ff_output
        x = self.norm3(x)
        
        return x


class Decoder(nn.Module):
    
    def __init__(self, vocab_size, d_model,  N, heads, dropout):
        super().__init__()

        self.N = N
        self.embed = Embedder(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model, max_seq_len=80, dropout=dropout)
        self.layers = [DecoderLayer(d_model, heads, dropout) for _ in range(N)]
        self.norm = LayerNorm(d_model)

    def forward(self, trg, e_outputs, src_mask, trg_mask):
        x = self.embed(trg)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, e_outputs, src_mask, trg_mask)
        x = self.norm(x)
        
        return x


class EncoderLayer(nn.Module):
    
    def __init__(self, d_model, heads, dropout = 0.1):
        super().__init__()

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, heads, dropout=dropout)
        self.ff = FeedForward(d_model, d_ff=2048, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x, mask):
        # multi-head attention
        attn_output = self.attn(x, x, x, mask)
        attn_output = self.dropout1(attn_output)
        # add & norm
        x = x + attn_output
        x = self.norm1(x)

        # feed forward
        ff_output = self.ff(x)
        ff_output = self.dropout2(ff_output)
        # add & norm
        x = x + ff_output
        x = self.norm2(x)

        return x


class Encoder(nn.Module):
    
    def __init__(self, vocab_size, d_model, N, heads, dropout):
        super().__init__()

        self.N = N
        self.embed = Embedder(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model, max_seq_len=80, dropout=dropout)
        self.layers = [EncoderLayer(d_model, heads, dropout) for _ in range(N)]
        self.norm = LayerNorm(d_model)
    
    def forward(self, src, mask):
        x = self.embed(src)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, mask)
        x = self.norm(x)
        
        return x


class Trnasformer(nn.Module):

    def __init__(self, src_vocab, trg_vocab, d_model, N, heads, dropout):
        super().__init__()

        self.encoder = Encoder(src_vocab, d_model, N, heads, dropout)
        self.decoder = Decoder(trg_vocab, d_model, N, heads, dropout)
        self.out = nn.Linear(d_model, trg_vocab)

    def forward(self, src, trg, src_mask, trg_mask):
        e_outputs = self.encoder(src, src_mask)
        d_output = self.decoder(trg, e_outputs, src_mask, trg_mask)
        output = self.out(d_output)
        
        return output




# 测试代码 main 函数
def main():
    import time
    import numpy as np
    # ------------------------------
    # Postion Embedding test
    # ------------------------------
    # params
    d_model = 2
    max_seq_len = 3
    
    # positional encoding
    pe = torch.zeros(max_seq_len, d_model)
    for pos in range(max_seq_len):
        for i in range(0, d_model, 2):
            pe[pos, i] = math.sin(pos / (10000 ** (i / d_model)))
            pe[pos, i + 1] = math.cos(pos / (10000 ** (i / d_model)))
    print(pe)
    print(pe.size())
    
    pe = pe.unsqueeze(0)
    print(pe)
    print(pe.size())
    
    x = torch.ones(1, max_seq_len, d_model)
    print(f"x size: {x.size(1)}")
    
    x = x + pe[:, :x.size(1)]
    print(f"x: \n{x}")
    
    print(pe[:, :x.size(1)])
    print(pe[:, :x.size(1), :])
    # ------------------------------
    # 
    # ------------------------------
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
                    print(f"Epoch: {epoch+1}, Step: {i+1}, Loss: {loss_avg}, Time: {(time.time() - start) // 60}, {time.time() - temp} per {print_every}")
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

if __name__ == "__main__":
    main()
