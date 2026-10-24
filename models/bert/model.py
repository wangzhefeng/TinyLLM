# -*- coding: utf-8 -*-

# ***************************************************
# * File        : model.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-10-14
# * Version     : 2.0.101800
# * Description : 从零实现 BERT：Embedding / Encoder / Pooler / MLM+NSP。
#                 v2 修正（对照标准 BERT）：
#                   1. 多头注意力按 hidden 维切分 head，加 1/sqrt(d_k) 缩放、padding mask、输出投影；
#                   2. Add & Norm 为标准残差 + LayerNorm（Post-LN）；
#                   3. FFN 激活函数 ReLU -> GELU；
#                   4. MLM 头补 transform（Dense+GELU+LayerNorm），decoder 与词嵌入权重共享。
# * Link        : https://arxiv.org/abs/1810.04805
# * Requirement : torch
# ***************************************************

# python libraries
from pathlib import Path
import math

import torch
import torch.nn as nn

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


# -----------------------------------------------------------------------------
# BertEncoder 组件
# -----------------------------------------------------------------------------
class MultiHeadAttention(nn.Module):
    """
    标准多头自注意力：按 hidden 维切分 head，scores / sqrt(head_dim)，
    支持 padding mask（mask 中 1 表示保留，0 表示屏蔽），含输出投影。
    """

    def __init__(self, hidden_size, head_num, dropout_prob=0.0):
        super().__init__()
        assert hidden_size % head_num == 0, "hidden_size 必须整除 head_num"
        self.head_num = head_num
        self.head_dim = hidden_size // head_num
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x, attention_mask=None):
        # x: (batch, seq_len, hidden)；attention_mask: (batch, 1, 1, seq_len)，1=keep 0=mask
        batch, seq_len, hidden = x.shape
        # (batch, seq_len, hidden) -> (batch, head_num, seq_len, head_dim)
        q = self.query(x).view(batch, seq_len, self.head_num, self.head_dim).transpose(1, 2)
        k = self.key(x).view(batch, seq_len, self.head_num, self.head_dim).transpose(1, 2)
        v = self.value(x).view(batch, seq_len, self.head_num, self.head_dim).transpose(1, 2)
        # scaled dot-product: (batch, head_num, seq_len, seq_len)
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            # 用 -1e9 而非 -inf，避免整行被 mask 时 softmax 产生 NaN
            scores = scores.masked_fill(attention_mask == 0, -1e9)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        # (batch, head_num, seq_len, head_dim) -> (batch, seq_len, hidden)
        out = (attn @ v).transpose(1, 2).reshape(batch, seq_len, hidden)
        return self.out_proj(out)


class AddNorm(nn.Module):
    """
    标准 Add & Norm（Post-LN）：LayerNorm(x + dropout(sublayer_out))
    """

    def __init__(self, hidden_size, dropout_prob=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x, sublayer_out):
        return self.norm(x + self.dropout(sublayer_out))


class FeedForward(nn.Module):
    """
    前馈网络：Linear -> GELU -> Linear（BERT 标准激活为 GELU）
    """

    def __init__(self, hidden_size, feed_num):
        super().__init__()
        self.linear1 = nn.Linear(hidden_size, feed_num)
        self.activation = nn.GELU()
        self.linear2 = nn.Linear(feed_num, hidden_size)

    def forward(self, x):
        return self.linear2(self.activation(self.linear1(x)))


class BertEncoder(nn.Module):
    def __init__(self, hidden_size, feed_num, head_num, dropout_prob=0.0):
        super().__init__()
        self.multi_head_att = MultiHeadAttention(hidden_size, head_num, dropout_prob)
        self.add_norm1 = AddNorm(hidden_size, dropout_prob)
        self.feed_forward = FeedForward(hidden_size, feed_num)
        self.add_norm2 = AddNorm(hidden_size, dropout_prob)

    def forward(self, x, attention_mask=None):
        x = self.add_norm1(x, self.multi_head_att(x, attention_mask))
        x = self.add_norm2(x, self.feed_forward(x))
        return x


# -----------------------------------------------------------------------------
# BertModel: 负责实现 BERT 模型的核心架构
# -----------------------------------------------------------------------------
class BertEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 词嵌入：用于将词汇索引映射到嵌入向量
        self.word_embeddings = nn.Embedding(config["vocab_size"], config["hidden_size"], padding_idx=config.get("pad_token_id", 0))
        # 位置嵌入: 用于将位置索引映射到位置嵌入向量，以捕获单词在句子中的位置信息
        self.position_embeddings = nn.Embedding(config["max_position_embeddings"], config["hidden_size"])
        # 段嵌入: 用于将令牌类型索引映射到类型嵌入向量
        self.token_type_embeddings = nn.Embedding(config["type_vocab_size"], config["hidden_size"])

        self.LayerNorm = nn.LayerNorm(config["hidden_size"])
        self.dropout = nn.Dropout(config["hidden_dropout_prob"])

    def forward(self, batch_idx, batch_seg_idx):
        # word embedding
        word_emb = self.word_embeddings(batch_idx)
        # position embedding: only use the actual sequence length, not max_position_embeddings
        pos_idx = torch.arange(0, batch_idx.shape[1], device=batch_idx.device)
        pos_idx = pos_idx.unsqueeze(0).expand(batch_idx.shape[0], -1)
        pos_emb = self.position_embeddings(pos_idx)
        # segment embedding
        token_emb = self.token_type_embeddings(batch_seg_idx)
        # embedding
        emb = word_emb + pos_emb + token_emb
        emb = self.LayerNorm(emb)
        emb = self.dropout(emb)

        return emb


class BertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config["hidden_size"], config["hidden_size"])
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertModel(nn.Module):
    """
    BERT是一种预训练深度双向Transformer模型，用于自然语言处理任务
    """
    def __init__(self, config):
        super().__init__()
        self.pad_token_id = config.get("pad_token_id", 0)
        # 将输入的索引转换为对应的嵌入表示，包括词嵌入、位置嵌入和段嵌入
        self.embedding = BertEmbedding(config)
        # 由多个 Transformer 编码器层组成，这些编码器层堆叠在一起，形成了 BERT 的主体结构。
        # 每个编码器层都包含自注意力机制和前馈网络，能够处理序列数据并捕捉长距离依赖关系
        self.bert_layers = nn.ModuleList([
            BertEncoder(config["hidden_size"], config["feed_num"], config["head_num"], config["hidden_dropout_prob"])
            for _ in range(config["layer_num"])
        ])
        # 对编码层的输出进行聚合，得到整个序列的表示，这通常用于分类任务
        self.pool = BertPooler(config)

    def forward(self, batch_idx, batch_seg_idx):
        # padding mask：(batch, 1, 1, seq_len)，PAD 位置为 0，注意力中屏蔽
        attention_mask = (batch_idx != self.pad_token_id).unsqueeze(1).unsqueeze(2)
        x = self.embedding(batch_idx, batch_seg_idx)
        for layer in self.bert_layers:
            x = layer(x, attention_mask)
        bert_out = self.pool(x)

        return x, bert_out


# -----------------------------------------------------------------------------
# MLM Head: transform（Dense+GELU+LayerNorm）+ decoder（与词嵌入共享权重）
# -----------------------------------------------------------------------------
class MaskedLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_size = config["hidden_size"]
        vocab_size = config["vocab_size"]
        self.transform = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.GELU()
        self.LayerNorm = nn.LayerNorm(hidden_size)
        # decoder 权重与 word_embeddings 共享（在 Model 中绑定），bias 独立
        self.decoder = nn.Linear(hidden_size, vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(vocab_size))

    def forward(self, hidden_states):
        x = self.transform(hidden_states)
        x = self.activation(x)
        x = self.LayerNorm(x)
        return self.decoder(x) + self.bias


# -----------------------------------------------------------------------------
# Model: 在 BertModel 的基础上添加了预训练任务的输出层和损失函数
# -----------------------------------------------------------------------------
class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Encoder layer
        self.bert = BertModel(config)
        # Masked Language Model(MLM) head
        self.cls_mask = MaskedLMHead(config)
        # MLM decoder 与词嵌入权重共享（标准 BERT 做法）
        self.cls_mask.decoder.weight = self.bert.embedding.word_embeddings.weight
        # Next Sentence Prediction(NSP) linear layer
        self.cls_next = nn.Linear(config["hidden_size"], 2)
        # BERT 标准参数初始化（initializer_range=0.02）。
        # 必须显式初始化：词嵌入与 MLM decoder 权重共享后，若沿用 nn.Embedding 默认的 N(0,1) 初始化，
        # logits = h @ W_e^T 的方差过大（约 ||h||^2 量级），softmax 过度自信，初始 MLM loss 会远高于 ln(vocab_size)
        self.initializer_range = config.get("initializer_range", 0.02)
        self.apply(self._init_weights)
        # Masked Language Model(MLM) loss layer
        self.loss_fun_mask = nn.CrossEntropyLoss(ignore_index=config.get("pad_token_id", 0))
        # Next Sentence Prediction(NSP) loss layer
        self.loss_fun_next = nn.CrossEntropyLoss()

    def _init_weights(self, module):
        """BERT 标准初始化：Linear/Embedding ~ N(0, initializer_range)，bias 为 0，LayerNorm 为 1/0"""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.initializer_range)
            if module.padding_idx is not None:
                nn.init.zeros_(module.weight[module.padding_idx])
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, batch_idx, batch_seg_idx, batch_mask_val=None, batch_label=None, return_preds: bool=False):
        # Encoder output
        bert_out1, bert_out2 = self.bert(batch_idx, batch_seg_idx)
        # MLM layer pred out
        pre_mask = self.cls_mask(bert_out1)
        # NSP layer pred out
        pre_next = self.cls_next(bert_out2)
        # Loss
        if (batch_mask_val is not None) and (batch_label is not None):
            # Masked Language Model(MLM)
            loss_mask = self.loss_fun_mask(pre_mask.reshape(-1, pre_mask.shape[-1]), batch_mask_val.reshape(-1))
            # Next Sentence Prediction(NSP)
            loss_next = self.loss_fun_next(pre_next, batch_label)
            # Loss
            loss = loss_mask + loss_next
            if return_preds:
                # 单次前向同时返回预测结果，评估时无需第二次前向
                return loss, torch.argmax(pre_mask, dim=-1), torch.argmax(pre_next, dim=-1)
            return loss
        else:
            # return pred out
            return torch.argmax(pre_mask, dim=-1), torch.argmax(pre_next, dim=-1)




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
