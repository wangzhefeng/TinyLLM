# -*- coding: utf-8 -*-

# ***************************************************
# * File        : tiny_transformer.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2024-09-24
# * Version     : 1.0.092404
# * Description : description
# * Link        : https://github.com/datawhalechina/tiny-universe/tree/main/content/TinyTransformer
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
import math
import inspect

import torch
import torch.nn as nn
from torch.nn import functional as F

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


def attention(q, k, v, dropout_module = None, is_causal = False, mask = None):
    """
    注意力计算函数
    """
    # 计算 QK^T / sqrt(d_k)，维度为 (B, nh, T, hs)×(B, nh, hs, T)->(B, nh, T, T)
    att = (q @ k.transpose(-2, -1) * (1.0 / math.sqrt(k.size(-1))))
    # 如果是解码器的 Casual LM，需要 mask 掉右上角的元素
    if is_causal:
        # 这里截取到序列长度，因为有些序列可能比 block_size 短
        att = att.masked_fill(mask[:, :, :k.size(-2), :k.size(-2)] == 0, float("-inf"))
    # 计算 softmax，维度为 (B, nh, T, T)
    att = F.softmax(att, dim = -1)
    # Attention Dropout
    att = dropout_module(att)
    # V.Score, 维度为 (B, nh, T, T)×(B, nh, T, hs)->(B, nh, T, hs)
    y = att @ v
    
    return y


class Embedding(nn.Module):
    """
    词向量模块
    input and output Embedding block
    """

    def __init__(self, config) -> None:
        super().__init__()
        self.embd = nn.Embedding(config.vocab_size, config.n_embd)
    
    def forward(self, x):
        x = self.embd(x)
        
        return x


class Dropout(nn.Module):
    """
    Dropout
    """

    def __init__(self, config) -> None:
        super().__init__()
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x):
        x = self.dropout(x)
        return x


class PositionalEncoding(nn.Module):
    """
    位置编码模块
    
    在输入(input -> Embedding)上加入了位置编码
    """

    def __init__(self, config):
        super().__init__()

        # Dropout 层
        self.dropout = nn.Dropout(p = config.dropout)
        
        # Position Embeding 层
        pe = torch.zeros(config.block_size, config.n_embd).float()
        pe.requires_grad = False
        # pos
        position = torch.arange(0, config.block_size, dtype=torch.float32).unsqueeze(1)
        # 1 / 10000^{2i/d_model}
        div_term = torch.exp(torch.arange(0, config.n_embd, 2).float() * -(math.log(10000.0) / config.n_embd))
        
        # apply sine to even indices: pe(pos, 2i) = sin(pos / 10000^{2i/d_model})
        pe[:, 0::2] = torch.sin(position * div_term)
        # apply cosine to odd indices: pe(pos, 2i+1) = cos(pos / 10000^{2i/d_model})
        pe[:, 1::2] = torch.cos(position * div_term)

        # add batch dimension
        pe = pe.unsqueeze(0)

        # register the positional encoding as a buffer (not a parameter)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        add positional encoding to the input tensor
        """
        # x.shape: [batch_size, seq_len, d_model]
        x = x + self.pe[:, :x.size(1)]
        # dropout
        x = self.dropout(x)

        return x


class MultiHeadAttention(nn.Module):
    """
    多头注意力计算模块
    """
    
    def __init__(self, config, is_causal = False):
        """
        config (_type_): 配置对象
        is_causal (bool, optional): 是否为 Masked Multi-Head Attention. Defaults to False.
        """
        super().__init__()
        
        # 隐藏层维度必须是头数的整数倍
        assert (config.n_embd % config.n_head == 0), "n_embd must be divisible by n_heads"

        # Wq, Wk, Wv 参数矩阵(每个参数矩阵为 n_embd×n_embd)
        self.c_attns = nn.ModuleList([
            nn.Linear(config.n_embd, config.n_embd, bias = config.bias)
            for _ in range(3)
        ])
        
        # 输出的线性层(维度为 n_embd×n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias = config.bias)
        
        # 注意力的 dropout
        self.attn_dropout = nn.Dropout(config.dropout)
        # 残差连接的 dropout
        self.resid_dropout = nn.Dropout(config.dropout)
        
        # 头数
        self.n_head = config.n_head
        # 隐藏层维度
        self.n_embd = config.n_embd
        # dropout 概率
        self.dropout = config.dropout
        # 是否是解码器的 Casual LM
        self.is_causal = is_causal
        
        # 判断是否使用 Flash Attention
        # (Pytorch2.0 支持，即判断 torch.nn.functional.scaled_dot_product_attention 是否存在)
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        # 如果不使用 Flash Attention，打印一个警告
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch>=2.0")
            # 如果自己实现 MHSA，需要一个 casual mask，确保 attention 只能作用在输入序列的左边
            # 此处使用 register_buffer 注册一个 bias 属性
            # bias 是一个上三角矩阵，维度为 1×1×block_size×block_size, block_size 为序列最大长度
            self.register_buffer(
                "bias", 
                torch.tril(
                    torch.ones(config.block_size, config.block_size)
                         .view(1, 1, config.block_size, config.block_size)
                )
            )
    
    def forward(self, query, key, value):
        """
        输入为 query、key、value，维度为 (B, seq_len, n_embd)
        """
        # batch_size, sequence length, embedding dimensionality(n_embd)
        B, T, C = query.size()
        # 计算 Q、K、V，输入通过参数矩阵层，维度为 (B, T, n_embd)×(n_embd, n_embd)->(B, T, n_embd)
        q, k, v = [self.c_attns[i](x) for i, x in zip(range(3), (query, key, value))]
        # 将 Q、K、V 拆分成多头，维度为 (B, T, n_head, C//n_head)，然后交换维度，变成 (B, n_head, T, C//n_head)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        # 注意力计算
        if self.flash:
            # 直接使用 Flash Attention
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, 
                attn_mask = None, 
                dropout_p = self.dropout if self.training else 0, 
                is_causal = self.is_causal,
            )
        else:
            # 手动实现注意力计算
            y = attention(
                q, k, v, 
                dropout_module = self.attn_dropout, 
                is_causal = self.is_causal, 
                mask = self.bias
            )
        # 将多头的结果拼接起来，先交换维度为 (B, T, n_head, C//n_head)，再拼接成 (B, T, n_head*C//n_head)
        # 使用 contigonous() 函数保证内存是连续的，否则会报错
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        # 经过输出层计算，维度为 (B, T, C)，再经过线性层残差连接
        y = self.resid_dropout(self.c_proj(y))

        return y


class FeedForward(nn.Module):
    """
    全连接模块
    """
    
    def __init__(self, config):
        """
        Transformer 的全连接模块有两个线性层，中间加了一个 RELU 激活函数
        """
        super().__init__() 
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias = config.bias)
        self.relu = nn.ReLU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias = config.bias)
        # self.dropout = nn.Dropout(config.dropout)
        self.dropout = Dropout(config)
    
    def forward(self, x):
        x = self.c_fc(x)
        x = self.relu(x)
        x = self.c_proj(x)
        x = self.dropout(x)

        return x


class LayerNorm(nn.Module):
    """
    层规范化模块
        - 在 PyTorch 的 LayerNorm 基础上添加了偏置，
          因为 PyTorch 的 LayerNorm 不支持偏置为 None
    """
    
    def __init__(self, ndim, bias) -> None:
        """
        初始化参数和偏置

        Args:
            ndim (_type_): _description_
            bias (_type_): _description_
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
    
    def forward(self, input):
        """
        直接调用 Pytorch 的 LayerNorm
        """
        return F.layer_norm(
            input = input,
            normalized_shape = self.weight.shape,
            weight = self.weight,
            bias = self.bias,
            eps = 1e-5
        )


class EncoderLayer(nn.Module):
    """
    Encoder Layer
    """
    
    def __init__(self, config):
        """
        一个 Encoder Layer 中有一个 Multi-Head Attention，Encoder 不需要掩码，传入 is_causal=Fasle
        一个 Encoder Layer 中有一个 Feed Forward
        一个 Encoder Layer 中有两个 LayerNorm，分别在 Attention 之前和 FeedForward 之前
        """
        super().__init__() 
        self.ln_1 = LayerNorm(config.n_embd, bias = config.bias) 
        self.attn = MultiHeadAttention(config, is_causal = False)
        self.ln_2 = LayerNorm(config.n_embd, bias = config.bias)
        self.mlp = FeedForward(config)
    
    def forward(self, x):
        # LayerNorm
        x = self.ln_1(x)
        # Multi-Head Attention -> Add & Norm 
        x = x + self.attn(x, x, x)  # Encoder 使用 Self-Attention，所以 Q、K、V 都是 x
        x = self.ln_2(x)
        # Feed Forward -> Add & Norm(LayerNorm 在 Encoder Block 中)
        x = x + self.mlp(x)
        
        return x


class Encoder(nn.Module):
    """
    Encoder Block
    """
    
    def __init__(self, config):
        """
        一个 Encoder 由 N 个 Encoder Layer 组成
        """
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(config) for _ in range(config.n_layer)
        ])
        self.norm = LayerNorm(config.n_embd, bias = config.bias)
    
    def forward(self, x):
        """
        分别通过 N 层 Encoder Layer 和 一个 Layer Norm 层
        """
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        
        return x


class DecoderLayer(nn.Module):
    """
    Decoder Layer
    """

    def __init__(self, config):
        """
        一个 Decoder Layer 中有一个 Masked Multi-Head Attention，Decoder 的第一部分是 Mask Attention，传入 is_causal=True
        一个 Decoder Layer 中有一个 Multi-Head Attention，Deocder 的第二部分是类似 Encoder 的 Attention，传入 is_causal=False
        一个 Decoder Layer 中有一个 Feed Forward，Decoder 第三部分是 FeedForward
        一个 Decoder Layer 中有三个 LayerNorm，分别在 Mask Attention 之前、Self-Attention 之前和 FeedForward 之前
        """
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias = config.bias) 
        self.m_attn = MultiHeadAttention(config, is_causal = True)
        self.ln_2 = LayerNorm(config.n_embd, bias = config.bias) 
        self.attn = MultiHeadAttention(config, is_causal = False)
        self.ln_3 = LayerNorm(config.n_embd, bias = config.bias) 
        self.mlp = FeedForward(config)

    def forward(self, x, enc_out):
        # LayerNorm
        x = self.ln_1(x)
        # Masked Multi-Head Attention -> Add & Norm
        ## 第一部分是一个 Mask Self Attention，Q、K、V 都是 x
        x = x + self.m_attn(x, x, x)
        x = self.ln_2(x)
        # Multi-Head Attention -> Add & Norm 
        ## 第二部分是一个类似于 Encoder 的 Attention，Q 是 x，K、V 是 Encoder 的输出
        x = x + self.attn(x, enc_out, enc_out)
        x = self.ln_3(x)
        # Feed Forward -> Add & Norm(LayerNorm 在 Decoder Block 中)
        x = x + self.mlp(x)
        
        return x


class Decoder(nn.Module):
    """
    Decoder Block
    """

    def __init__(self, config):
        """
        一个 Decoder 由 N 个 Decoder Layer 组成
        """
        super().__init__() 
        self.layers = nn.ModuleList([
            DecoderLayer(config) for _ in range(config.n_layer)
        ])
        self.norm = LayerNorm(config.n_embd, bias = config.bias)
    
    def forward(self, x, enc_out):
        """
        将输入(和 mask)分别通过 N 层 Decoder Layer 和 一个 Layer Norm 层
        """
        for layer in self.layers:
            x = layer(x, enc_out)
        x = self.norm(x)
        
        return x


class Linear(nn.Module):
    """
    线性层
    """

    def __init__(self, config):
        super().__init__()
        self.linear = nn.Linear(config.n_embd, config.vocab_size, bias = False)

    def forward(self, x):
        x = self.linear(x)
        return x


# TODO 未使用
class Softmax(nn.Module):
    """
    Softmax 层
    """

    def __init__(self, logits):
        super().__init__()
        self.softmax = F.softmax(logits, dim = -1)
    
    def forward(self):
        return self.softmax


class Transformer(nn.Module):
    """
    Transformer 模型
    """

    def __init__(self, config):
        super().__init__()
        
        # 必须输入: 词表大小和 block size
        assert config.vocab_size is not None
        assert config.block_size is not None
        # ------------------------------
        # 参数配置
        # ------------------------------
        self.config = config
        # ------------------------------
        # Transformer
        # ------------------------------
        self.transformer = nn.ModuleDict(dict(
            # wte = nn.Embedding(config.vocab_size, config.n_embd),
            wte = Embedding(config),
            wpe = PositionalEncoding(config),
            # drop = nn.Dropout(config.dropout),
            drop = Dropout(config),
            encoder = Encoder(config),
            decoder = Decoder(config),
        ))
        # ------------------------------
        # 最后的线性层：输入是 n_embd，输出是词表大小(vocab_size)
        # ------------------------------
        # self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias = False)
        self.lm_head = Linear(config)
        # ------------------------------
        # 初始化所有的权重 
        # ------------------------------ 
        self.apply(self._init_weights)
        # ------------------------------
        # 计算所有参数的数量
        # ------------------------------
        print(f"number of parameters: {self._get_num_params() / 1e6:.2f}M")
    
    def _get_num_params(self, non_embedding = False) -> int:
        """
        统计所有参数的数量

        Args:
            non_embedding (bool, optional): 是否统计 embedding 的参数. Defaults to False.

        Returns:
            (int): 所有参数的数量
        """
        # 统计所有模型层参数
        n_params = sum(p.numel() for p in self.parameters())
        # 如果不统计 embedding 的参数，就减去
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        
        return n_params

    def _init_weights(self, module):
        """
        初始化权重：线性层和 Embedding 层初始化为正态分布
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean = 0.0, std = 0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean = 0.0, std = 0.02)

    def forward(self, idx, targets = None):
        """
        前向计算函数
        
        Args:
            idx (_type_): 输入，维度 (batch size, sequence lenth)
            targets (_type_, optional): 目标序列，用于计算 loss. Defaults to None.
        """
        # device
        device = idx.device
        # batch_size, sequence_length
        batch_size, seq_len = idx.size()
        # sequence_length <= block_size
        assert seq_len <= self.config.block_size, f"不能计算该序列，该序列长度为 {seq_len}, 最大序列长度只有 {self.config.block_size}"
        
        # ------------------------------
        # 通过 self.transformer
        # ------------------------------
        ## 通过：Embedding Layer (得到的维度是 (batch size, sequence length, vocab_size, n_embd))
        print(f"x size: {idx.size()}")
        tok_emb = self.transformer.wte(idx)
        print(f"x after wte: {tok_emb.size()}")
        ## 通过：Positional Encoding (得到的维度是 (batch size, sequence length, vocab_size, n_embd))
        pos_emb = self.transformer.wpe(tok_emb)   
        x = self.transformer.drop(pos_emb)  # 进行：dropout
        print(f"x after wpe: {x.size()}")

        ## 通过：Encoder 
        enc_out = self.transformer.encoder(x)
        print(f"enc_out size: {enc_out.size()}")

        ## 通过：Decoder
        x = self.transformer.decoder(x, enc_out)
        print(f"x after decoder: {x.size()}")

        # ------------------------------
        # 训练阶段、推理阶段
        # ------------------------------
        # 训练阶段：如果给了 targets，就计算 loss
        if targets is not None: 
            # 1.先通过最后的 Linear 层(维度为 (batch size, sequence length, vocab size))
            logits = self.lm_head(x)
            # 2.再跟 targets 计算交叉熵
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index = -1)
        # 推理阶段：只需要 logits, loss 为 None(取 -1 是只取序列中的最后一个作为输出)
        else: 
            logits = self.lm_head(x[:, [-1], :])
            loss = None
        
        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate: float, betas, device_type):
        """
        配置优化器

        Args:
            weight_decay (_type_): 权重衰减系数
            learning_rate (_type_): 学习率
            betas (_type_): AdamW 的 betas
            device_type (_type_): 设备类型
        """
        # 获取所有命名参数
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # 过滤掉不需要更新的参数
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        # 参数根据维度分为两组
        ## 维度大于等于 2 的参数（通常是权重）会应用权重衰减
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        ## 维度小于 2 的参数（通常是偏置和层归一化参数）不会应用权重衰减
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = {
            {
                "params": decay_params, 
                "weight_decay": weight_decay
            },
            {
                "params": nodecay_params, 
                "weight_decay": 0.0
            },
        }
        # 打印以下参数数量
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"应用权重衰减的层数: {len(decay_params)}; 总参数量为: {num_decay_params:,}")
        print(f"不应用权重衰减的层数: {len(nodecay_params)}; 总参数量为: {num_nodecay_params:,}")

        # 检查 torch.optim.AdamW 是否支持融合版本（fused version），这是针对 CUDA 设备优化的版本。
        # 如果可用且 device_type 为 "cuda"，则使用融合版本。
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        extra_args = dict(fused = True) if use_fused else dict()

        # 创建优化器
        optimizer = torch.optim.AdamW(optim_groups, lr = learning_rate, betas = betas, **extra_args)
        print(f"是否使用 fused AdamW: {use_fused}")
        
        return optimizer

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature = 1.0, top_k = None):
        """
        模型推理

        Args:
            idx (_type_): 输入, 维度为 (batch size, sequence length)
            max_new_tokens (_type_): 最大生成的 token 数量，即按序推理 max_new_tokens 次
            temperature (float, optional): 温度参数. Defaults to 1.0.
            top_k (_type_, optional): _description_. Defaults to None.
        """
        for _ in range(max_new_tokens):
            # 如果输入序列太长，需要将它截断到 block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]

            # 前向计算，得到 logits，维度为 (batch size, sequence length, vocab size)
            logits, _ = self(idx_cond)
            # 使用最后一个 token 的 logits 作为当前输出，除以温度系数控制其多样性
            logits = logits[:, -1, :] / temperature

            # 如果使用 Top K 采样，将 logits 中除了 top_k 个元素的概率置为 0
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            
            # 对输出结果进行 Softmax
            probs = F.softmax(logits, dim = -1)
            # 对结果概率进行采样
            idx_next = torch.multinomial(probs, num_samples = 1)

            # 将输出结果拼接到输入序列后面，作为下一次的输入
            idx = torch.cat((idx, idx_next), dim = 1)
            # print(f"idx: {idx}")
        
        return idx




# 测试代码 main 函数
def main():
    import torch
    from dataclasses import dataclass

    # ------------------------------
    # 创建模型配置文件
    # ------------------------------ 
    print("*" * 80)
    @dataclass
    class TransformerConfig:
        block_size: int = 1024  # 序列的最大长度
        vocab_size: int = 50304  # 词表大小
        n_layer: int = 4  # Encoder, Deocder 层数
        n_head: int = 4  # 注意力头数量
        n_embd: int = 768  # Embedding 维度(d_model)
        dropout: float = 0.0
        bias: bool = True

    model_config = TransformerConfig(
        block_size = 12,
        vocab_size = 10,
        n_layer = 2,
        n_head = 4,
        n_embd = 16,
        dropout = 0.0,
        bias = True,
    )
    print(f"Model Config:\n{model_config}")
    # ------------------------------
    # 创建模型
    # ------------------------------
    print("*" * 80)
    print("Model info:")
    print("-" * 45)
    model = Transformer(model_config)
    # ------------------------------
    # 向前传递
    # ------------------------------
    idx = torch.randint(1, 10, (4, 8))
    print(f"Model input:\n{idx}")
    print(f"Model input size:\n{idx.size()}")
    print("-" * 45)
    print("Model forward:")
    logits, loss = model(idx, targets = None)
    print("-" * 45)
    print(f"logits:\n{logits} \nlogits size: {logits.size()}")
    print(f"loss: {loss}")
    # ------------------------------
    # 模型推理
    # ------------------------------ 
    print("*" * 80)
    print("Model Inference:")
    print("-" * 45)
    result = model.generate(idx, 3)
    print("-" * 45)
    print(f"generate result:\n{result}")
    print(f"generate result size: {result.size()}")

if __name__ == "__main__":
    main()
