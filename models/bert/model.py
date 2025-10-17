# -*- coding: utf-8 -*-

# ***************************************************
# * File        : model.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-10-14
# * Version     : 1.0.101417
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


# -----------------------------------------------------------------------------
# BertEncoder
# -----------------------------------------------------------------------------
class Multi_Head_Att(nn.Module):
    def __init__(self, hidden_num, head_num):
        super().__init__()
        self.Q = nn.Linear(hidden_num, hidden_num)
        self.K = nn.Linear(hidden_num, hidden_num)
        self.V = nn.Linear(hidden_num, hidden_num)

        self.head_num = head_num
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # 10 * 128 * 768
        batch, len_, hidden = x.shape
        x = x.reshape(batch, self.head_num, -1, hidden)
        q = self.Q(x) # 10 * 128 * 768
        k = self.K(x) # 10 * 128 * 768
        v = self.V(x) # 10 * 128 * 768
        score = self.softmax(q @ k.transpose(-2,-1))
        x = score @ v
        x = x.reshape(batch, len_, hidden)
        return x

class Add_Norm(nn.Module):
    def __init__(self, embedding_num):
        super().__init__()
        self.Add = nn.Linear(embedding_num, embedding_num)
        self.Norm = nn.LayerNorm(embedding_num)

    def forward(self, x):
        # B * Layer * emb
        x = self.Add(x)
        x = self.Norm(x)
        return x

class Feed_Forward(nn.Module):
    def __init__(self, embedding_num, feed_num):
        super().__init__()
        self.linear1 = nn.Linear(embedding_num, feed_num)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(feed_num, embedding_num)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

class BertEncoder(nn.Module):
    def __init__(self, hidden_size, feed_num, head_num):
        super().__init__()
        self.multi_head_att = Multi_Head_Att(hidden_size, head_num)  # 768 * 768
        self.add_norm1 = Add_Norm(hidden_size)
        self.feed_forward = Feed_Forward(hidden_size, feed_num)
        self.add_norm2 = Add_Norm(hidden_size)

    def forward(self,x): 
        multi_head_out = self.multi_head_att(x) 
        add_norm1_out = self.add_norm1(multi_head_out) 
        add_norm1_out = x + add_norm1_out

        feed_forward_out = self.feed_forward(add_norm1_out)  
        add_norm2_out = self.add_norm2(feed_forward_out) 
        add_norm2_out = add_norm1_out + add_norm2_out

        return add_norm2_out

# -----------------------------------------------------------------------------
# BertModel
# -----------------------------------------------------------------------------
class BertEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config["vocab_size"], config["hidden_size"])
        self.word_embeddings.weight.requires_grad = True

        self.position_embeddings = nn.Embedding(config["max_position_embeddings"], config["hidden_size"])
        self.position_embeddings.weight.requires_grad = True

        self.token_type_embeddings = nn.Embedding(config["type_vocab_size"], config["hidden_size"])
        self.token_type_embeddings.weight.requires_grad = True

        self.LayerNorm = nn.LayerNorm(config["hidden_size"])
        self.dropout = nn.Dropout(config["hidden_dropout_prob"])

    def forward(self, batch_idx, batch_seg_idx):
        # word embedding
        word_emb = self.word_embeddings(batch_idx)
        # position embedding
        pos_idx = torch.arange(0, self.position_embeddings.weight.data.shape[0], device=batch_idx.device)
        pos_idx = pos_idx.repeat(batch_idx.shape[0], 1)
        pos_emb = self.position_embeddings(pos_idx)
        # token embedding
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
    def __init__(self, config):
        super().__init__()
        self.embedding = BertEmbedding(config)
        # self.bert_layer = nn.Linear(config["hidden_size"],config["hidden_size"])
        self.bert_layer = nn.Sequential(
            *[BertEncoder(config["hidden_size"], config["feed_num"], config["head_num"]) 
              for i in range(config["layer_num"])]
        )
        self.pool = BertPooler(config)

    def forward(self, batch_idx, batch_seg_idx):
        x = self.embedding(batch_idx, batch_seg_idx)
        x = self.bert_layer(x)
        bert_out = self.pool(x)

        return x, bert_out

class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.bert = BertModel(config)
        self.cls_mask = nn.Linear(config["hidden_size"], config["vocab_size"])
        self.cls_next = nn.Linear(config["hidden_size"], 2)
        self.loss_fun_mask = nn.CrossEntropyLoss(ignore_index=0)
        self.loss_fun_next = nn.CrossEntropyLoss()

    def forward(self, batch_idx, batch_seg_idx, batch_mask_val=None, batch_label=None):
        bert_out1, bert_out2 = self.bert(batch_idx, batch_seg_idx)
        pre_mask = self.cls_mask(bert_out1)
        pre_next = self.cls_next(bert_out2)
        if (batch_mask_val is not None) and (batch_label is not None):
            loss_mask = self.loss_fun_mask(pre_mask.reshape(-1, pre_mask.shape[-1]), batch_mask_val.reshape(-1))
            loss_next = self.loss_fun_next(pre_next, batch_label)
            loss = loss_mask + loss_next
            return loss
        else:
            return torch.argmax(pre_mask, dim=-1), torch.argmax(pre_next, dim=-1)




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
