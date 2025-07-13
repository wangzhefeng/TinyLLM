# -*- coding: utf-8 -*-

# ***************************************************
# * File        : transformer_encdec.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-07-13
# * Version     : 1.0.071302
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
import time
import copy
import math
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F

# attention
from layers.attention import (
    MultiHeadAttention,
    MultiHeadAttentionRoPE,
    GroupedQueryAttention,
)
# feed forward
from layers.feed_forward import (
    FeedForwardReLU,
    FeedForwardGELU,
    FeedForwardSiLU,
)
# normization
from layers.rms_norm import RMSNorm
from layers.layer_norm import LayerNorm
# positional encoding
from layers.RoPE import precompute_rope_params, compute_rope
from layers.FixPE import PositionalEncoding
# word embedding
from layers.WoEmbed import Embeddings

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


ffn_type = {
    "relu": FeedForwardReLU,
    "gelu": FeedForwardGELU,
    "silu": FeedForwardSiLU,
}
mha_type = {
    "mha": MultiHeadAttention,
    "mha_rope": MultiHeadAttentionRoPE,
    "gqa": GroupedQueryAttention,
}
norm_type = {
    "rms": RMSNorm,
    "ln": LayerNorm,
    "ln_torch": nn.LayerNorm
}
woe_type = {
    "torch": nn.Embedding,
    "others": None,
}
wpe_type = {
    "fixed": PositionalEncoding,
    "rope": compute_rope,
}


# TODO
def clones(module, N):
    """
    Produce N identical layers
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


# TODO
class LayerNorm(nn.Module):
    """
    Construct a layernorm module
    """

    def __init__(self, features, eps=1e-6):
        super().__init__()

        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps
    
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        norm_x = self.a_2 * (x - mean) / (std + self.eps) + self.b_2

        return self.a_2 * norm_x + self.b_2


# TODO
class ResidualConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    
    def __init__(self, size, dropout):
        super().__init__()

        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, sublayer):
        """
        Apply residual connection to any sublayer with the same size.
        """
        norm_x = self.norm(x)
        sublayer_x = sublayer(norm_x)
        res_con_x = x + self.dropout(sublayer_x)

        return res_con_x


# TODO
class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, query, key, value, mask=None, dropout=None):
        # query and key of dimension d_k
        d_k = query.size(-1)
        # attention scores
        scores = query@key.transpose(-2, -1) / math.sqrt(d_k)
        # mask attention
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        # softmax
        p_attn = F.softmax(scores, dim=-1)
        # dropout
        if dropout is not None:
            p_attn = dropout(p_attn)
        
        return p_attn@value, p_attn


# TODO
class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention
    """
    
    def __init__(self, num_heads, d_model, dropout=0.1):
        super().__init__()

        assert d_model % num_heads == 0

        # assume d_value always equals d_key
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.linears = clones(
            nn.Linear(d_model, d_model), 4
        )
        # 记录 attention 矩阵结果
        self.attn = None
        self.dropout = nn.Dropout(dropout)
        self.attention = ScaledDotProductAttention()

    def forward(self, query, key, value, mask=None):
        # same mask applied to all h heads
        if mask is not None:
            mask = mask.unsqueeze(1)
        # number of batches
        n_batches = query.size(0)
        # do all the linear projections in batch from: d_model => num_heads x d_k
        query, key, value = [
            linear(x).view(n_batches, -1, self.num_heads, self.d_k).transpose(1, 2)
            for linear, x in zip(self.linears, (query, key, value))
        ]
        # apply attention on all the projected vectors in batch
        x, self.attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)
        # concat using a view
        x = (
            x \
            .transpose(1, 2) \
            .contiguous() \
            .view(n_batches, -1, self.num_heads * self.d_k)
        )
        # delete query, key, value
        del query, key, value
        # final linear
        output = self.linears[-1](x)

        return output


# TODO
class PositionwiseFeedForward(nn.Module):
    """
    Implements FFN equation
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()

        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        ffn_1 = F.relu(self.w_1(x))
        x = self.dropout(ffn_1)
        ffn_2 = self.w_2()

        return ffn_2


# TODO
def subsequent_mask(size):
    """
    Mask out subsequent positions
    """
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(
        torch.ones(attn_shape), 
        diagonal=1
    ).type(torch.uint8)
    print(subsequent_mask)

    return subsequent_mask == 0


class EncoderLayer(nn.Module):

    def __init__(self, 
                 ffn: str, mha: str, norm: str,
                 d_model: int, num_heads: int, d_ff: int, dropout: float=0.1):
        super().__init__()

        self.self_attn = mha_type[mha](d_model, num_heads)
        self.feed_forward = ffn_type[ffn](d_model, d_ff)
        self.norm1 = norm_type[norm](d_model)
        self.norm2 = norm_type[norm](d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # self-attention with residual connection and layer normalization
        attn_output = self.self_attn(x, x, x, mask)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)
        # feed forward with residual connection and layer normalization
        ff_output = self.feed_forward(x)
        x = x + self.dropout(ff_output)
        x = self.norm2(x)

        return x


class Encoder(nn.Module):
    """
    Core encoder is a stack of N layers
    """
    
    def __init__(self, woe: str, wpe: str,
                 vocab_size: int, d_model: int, num_heads: int, 
                 d_ff: int, num_layers: int, dropout: float=0.1):
        super().__init__()

        # params
        self.d_model = d_model
        # layers
        self.word_embedding = woe_type[woe](vocab_size, d_model)
        self.positional_encoding = wpe_type[wpe](d_model)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # embedding and positional encoding
        x = self.word_embedding(x) * math.sqrt(self.d_model)
        x = x + self.positional_encoding(x)
        x = self.dropout(x)
        # pass through encoder layers
        for layer in self.layers:
            x = layer(x, mask)
        
        return x


# TODO
class EncoderLayer(nn.Module):
    """
    Encoder is made up of self-attention and feed forward
    """

    def __init__(self, size, self_attn, feed_forward, dropout):
        super().__init__()

        self.size = size
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(
            ResidualConnection(size, dropout), 2
        )
    
    def forward(self, x, mask):
        x_attn = self.sublayer[0](
            x, 
            lambda x: self.self_attn(x, x, x, mask)
        )
        x_ffn = self.sublayer[1](
            x_attn, 
            self.feed_forward
        ) 

        return x_ffn


# TODO
class Encoder(nn.Module):
    """
    Core encoder is a stack of N layers
    """

    def __init__(self, layer, N):
        super().__init__()

        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
    
    def forward(self, x, mask):
        """
        Pass the input(and mask) through each layer in turn.
        """
        for layer in self.layers:
            x = layer(x, mask)
        
        return self.norm(x)


class DecoderLayer(nn.Module):
    
    def __init__(self, 
                 ffn: str, mha: str, norm: str,
                 d_model: int, num_heads: int, d_ff: int, dropout: float=0.1):
        super().__init__()

        self.self_attn = mha_type[mha](d_model, num_heads)
        self.cross_attn = mha_type[mha](d_model, num_heads)
        self.feed_forward = ffn_type[ffn](d_model, d_ff)
        self.norm1 = norm_type[norm](d_model)
        self.norm2 = norm_type[norm](d_model)
        self.norm3 = norm_type[norm](d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        # self-attention with residual connection and layer normalization
        self_attn_output = self.self_attn(x, x, x, tgt_mask)
        x = x + self.dropout(self_attn_output)
        x = self.norm1(x)
        # cross-attention with residual connection and layer normalization
        cross_attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = x + self.dropout(cross_attn_output)
        x = self.norm2(x)
        # feed forward with residual connection and layer normalization
        ff_output = self.feed_forward(x)
        x = x + self.dropout(ff_output)
        x = self.norm3(x)

        return x


class Decoder(nn.Module):
    
    def __init__(self, woe: str, wpe: str,
                 vocab_size: int, d_model: int, num_heads: int, 
                 d_ff: int, num_layers: int, dropout: float=0.1):
        super().__init__()

        # params
        self.d_model = d_model
        # layers
        self.word_embedding = woe_type[woe](vocab_size, d_model)
        self.positional_encoding = wpe_type[wpe](d_model)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        # embedding and positional encoding
        x = self.word_embedding(x) * math.sqrt(self.d_model)
        x = x + self.positional_encoding(x)
        x = self.dropout(x)
        # pass through decoder layers
        for layer in self.layers:
            x = layer(x, enc_output, src_mask, tgt_mask)
        
        return x


# TODO
class DecoderLayer(nn.Module):
    """
    Deocder is made of self-attention, src-attention, and feed forward
    """

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super().__init__()

        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(
            ResidualConnection(size, dropout), 3
        )
    
    def forward(self, x, memory, src_mask, tgt_mask):
        m = memory
        x_self_attn = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x_src_attn = self.sublayer[1](x_self_attn, lambda x: self.src_attn(x, m, m, src_mask))
        x_ffn = self.sublayer[2](x_src_attn, self.feed_forward)
        
        return x_ffn 


# TODO
class Decode(nn.Module):
    """
    Generic N layer decoder with masking
    """

    def __init__(self, layer, N):
        super().__init__()

        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
    
    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        
        return self.norm(x)


# TODO
class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """
    
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
    
    def forward(self, src, tgt, src_mask, tgt_mask):
        """
        Take in an process masked src and target sequences.

        Args:
            src (_type_): _description_
            tgt (_type_): _description_
            src_mask (_type_): _description_
            tgt_mask (_type_): _description_
        """
        enc_output = self.encoder(src, src_mask)
        dec_output = self.decoder(enc_output, src_mask, tgt, tgt_mask)

        return dec_output
    
    def encode(self, src, src_mask):
        enc_input = self.src_embed(src)
        enc_output = self.encoder(enc_input, src_mask)

        return enc_output
    
    def decode(self, memory, src_mask, tgt, tgt_mask):
        dec_input = self.tgt_embed(tgt)
        dec_output = self.decoder(dec_input, memory, src_mask, tgt_mask)

        return dec_output


# TODO
class Generator(nn.Module):
    """
    Define standard linear + softmax generation step.
    """

    def __init__(self, d_model, vocab_size):
        super().__init__()

        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        proj = self.proj(x)
        logit = F.log_softmax(proj, dim=-1)

        return logit


class DummyOptimizer(torch.optim.Optimizer):
    
    def __init__(self):
        self.param_groups = [{"lr": 0}]
        None
    
    def step(self):
        None
    
    def zero_grad(self, set_to_none=False):
        None


class DummyScheduler:
    
    def step(self):
        None


def make_model(src_vocab, tgt_vocab, 
               n_layers: int=6, d_model: int=512, 
               d_ff: int=2048, num_heads: int=8, 
               dropout: float=0.1):
    """
    Construct a model from hyperparameters

    Args:
        src_vocab (_type_): _description_
        tag_vocab (_type_): _description_
        n_layers (int, optional): _description_. Defaults to 6.
        d_model (int, optional): _description_. Defaults to 512.
        d_ff (int, optional): _description_. Defaults to 2048.
        num_heads (int, optional): _description_. Defaults to 8.
        dropout (float, optional): _description_. Defaults to 0.1.
    """
    c = copy.deepcopy
    mha = MultiHeadAttention(num_heads, d_model)
    pff = PositionwiseFeedForward(d_model, d_ff, dropout)
    wpe = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(mha), c(pff), dropout), n_layers),
        Decoder(DecoderLayer(d_model, c(mha), c(mha), c(pff), dropout), n_layers),
        nn.Sequential(Embeddings(d_model, src_vocab), c(wpe)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(wpe)),
        Generator(d_model, tgt_vocab)
    )
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return model


def inference():
    # model
    model = make_model(src_vocab=11, tgt_vocab=11, n_layers=2)
    model.eval()
    # src
    src = torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    # src mask
    src_mask = torch.ones(1, 1, 10)
    # memory
    memory = model.encode(src, src_mask)
    # TODO
    ys = torch.zeros(1, 1).type_as(src)
    for i in range(9):
        out = model.deocde(
            memory, 
            src_mask, 
            ys,
            subsequent_mask(ys.size(1)).type_as(src.data)
        )
        prob = model.generator()
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat([
            ys, 
            torch.empty(1, 1).type_as(src.data).fill_(next_word)
        ], dim=1)
    print(f"Untrained Model Prediction: {ys}")


def run_test():
    for _ in range(10):
        inference()


class Batch:
    """
    Object for holding a batch of data with mask during training.
    """

    def __init__(self, src, tgt=None, pad=2):  # 2 = <blank>
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if tgt is not None:
            self.tgt = tgt[:, :-1]
            self.tgt_y = tgt[:, 1:]
            self.tgt_mask = self.make_std_mask(self.tgt, pad)
            self.ntokens = (self.tgt_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(
            tgt_mask.data
        )
        return tgt_mask


class TrainState:
    """
    Track number of steps, examples, and tokens processed
    """
    step: int = 0  # Steps in the current epoch
    accum_step: int = 0  # Number of gradient accumulation steps
    samples: int = 0  # total # of examples used
    tokens: int = 0  # total # of tokens processed


def run_epoch(
    data_iter,
    model,
    loss_compute,
    optimizer,
    scheduler,
    mode="train",
    accum_iter=1,
    train_state=TrainState(),
):
    """Train a single epoch"""
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    n_accum = 0
    for i, batch in enumerate(data_iter):
        out = model.forward(
            batch.src, batch.tgt, batch.src_mask, batch.tgt_mask
        )
        loss, loss_node = loss_compute(out, batch.tgt_y, batch.ntokens)
        # loss_node = loss_node / accum_iter
        if mode == "train" or mode == "train+log":
            loss_node.backward()
            train_state.step += 1
            train_state.samples += batch.src.shape[0]
            train_state.tokens += batch.ntokens
            if i % accum_iter == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                n_accum += 1
                train_state.accum_step += 1
            scheduler.step()

        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        if i % 40 == 1 and (mode == "train" or mode == "train+log"):
            lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - start
            print(
                (
                    "Epoch Step: %6d | Accumulation Step: %3d | Loss: %6.2f "
                    + "| Tokens / Sec: %7.1f | Learning Rate: %6.1e"
                )
                % (i, n_accum, loss / batch.ntokens, tokens / elapsed, lr)
            )
            start = time.time()
            tokens = 0
        del loss
        del loss_node
    return total_loss / total_tokens, train_state




# 测试代码 main 函数
def main():
    # subsequent mask
    size = 5
    res = subsequent_mask(size)
    print(res)

    # model test
    run_test()

if __name__ == "__main__":
    main()
