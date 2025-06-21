# -*- coding: utf-8 -*-

# ***************************************************
# * File        : llama3.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2025-02-23
# * Version     : 0.1.022317
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import os
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)
import json


import torch

from tokenizer.llama3_8b_bpe import llama3_8b_tokenizer
from utils.log_util import logger

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


# ------------------------------
# 加载模型(一个网络层名称-tensor类型参数的字典)
# ------------------------------
model_path = Path(ROOT).joinpath(r"downloaded_models\llama_model\Meta-Llama-3-8B\original\consolidated.00.pth")
model = torch.load(model_path, map_location=torch.device("cpu"), weights_only=True)
logger.info(f"model layers names 20 ahead: {json.dumps(list(model.keys())[:20], indent=4)}")

# ------------------------------
# 加载模型配置文件
# ------------------------------
params_path = Path(ROOT).joinpath(r"downloaded_models\llama_model\Meta-Llama-3-8B\original\params.json")
with open(params_path, "r") as file:
    config = json.load(file)
logger.info(f"model config: {json.dumps(config, indent=4)}")

dim = config["dim"]
n_layers = config["n_layers"]
n_heads = config["n_heads"]
n_kv_heads = config["n_kv_heads"]
vocab_size = config["vocab_size"]
multiple_of = config["multiple_of"]
ffn_dim_multiplier = config["ffn_dim_multiplier"]
norm_eps = config["norm_eps"]
rope_theta = torch.tensor(config["rope_theta"])

# ------------------------------
# 将文本转换为 token id 序列
# ------------------------------
tokenzier = llama3_8b_tokenizer()

# ------------------------------
# 将 token id 序列转换为 embedding 嵌入向量
# ------------------------------
# 创建一个嵌入层网络，用于将离散的 token id 映射到连续的向量空间# 
embed_layer = torch.nn.Embedding(vocab_size, dim)
# 将嵌入层网络的参数替换为 llama3 中预训练好的参数值
embed_layer.weight.data.copy_(model["tok_embeddings.weight"])


# 测试代码 main 函数
def main():
    # prompt
    prompt_text = "the answer to the ultimate question of life, the universe, and everything is " 
    # token ids
    tokens_ids = [128000] + tokenzier.encode(prompt_text)
    tokens_ids = torch.tensor(tokens_ids)
    logger.info(f"tokens: \n{tokens_ids} \ntokens lenght: {len(tokens_ids)}")
    # 使用嵌入层网络，将输入的 token id 序列转换为向量表示
    # 嵌入层网络仅是基于 id 查字典来找到对应的向量，不涉及 token 间的交互([17] -> [17x4096]) 
    # 默认是 float32 全精度，这里换成半精度格式，降低内存占用
    token_embed_unnorm = embed_layer(tokens_ids).to(torch.bfloat16)
    logger.info(f"token_embed_unnorm.shape: {token_embed_unnorm.shape}")

if __name__ == "__main__":
    main()
