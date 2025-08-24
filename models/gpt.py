# -*- coding: utf-8 -*-

# ***************************************************
# * File        : gpt.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-08-17
# * Version     : 1.0.081723
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
from dataclasses import dataclass
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.tokenizers.tokenization import choose_tokenizer
from data_provider.data_loader import create_dataloader
from utils.device import device_setting
from utils.log_util import logger

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]

@dataclass
class GPTConfig:
    # data path
    data_path="EleutherAI/wikitext_document_level"
    data_file="wikitext-2-raw-v1"
    # data config
    seq_len: int = 4
    # tokenizer config
    tokenizer_model = "tiktoken_gpt2_bpe"
    tokenizer = choose_tokenizer(tokenizer_model)
    vocab_size: int = tokenizer.n_vocab
    # model config
    n_layer: int = 2
    n_heads: int = 3
    embed_dim: int = 6





# 测试代码 main 函数
def main():
    # device
    device = device_setting(verbose=True)

    # config
    gpt_config = GPTConfig()
    logger.info(f"gpt_config: \n{gpt_config}")

    # token embedding
    token_embedding = nn.Embedding(gpt_config.vocab_size, gpt_config.embed_dim)

if __name__ == "__main__":
    main()
