# -*- coding: utf-8 -*-

# ***************************************************
# * File        : opeai_gpt2_models.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2025-03-02
# * Version     : 0.1.030223
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
from transformers import GPT2Model

from model_load.openai_gpt2_weights_load_hf import load_weights
from utils.argsparser_tools import DotDict

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


# huggingface allowed model names
gpt2_model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

# huggingface gpt2 model
gpt2_huggingface_models = {
    "gpt2-small (124M)": "openai-community/gpt2",
    "gpt2-medium (355M)": "openai-community/gpt2-medium",
    "gpt2-large (774M)": "openai-community/gpt2-large",
    "gpt2-xl (1558M)": "openai-community/gpt2-xl"
}


def load_pretrained_gpt2_model(cfgs, model_cls):
    """
    initializing a model with pretrained weights
    """
    # Loading pretrained LLM
    base_config = {
        "vocab_size": cfgs.vocab_size,          # Vocabulary size: 50257
        "context_length": cfgs.context_length,  # Context length: 1024
        "dropout": cfgs.dropout,                # Dropout rate: 0.0
        "qkv_bias": cfgs.qkv_bias,              # Query-key-value bias: True
    }
    base_config.update(gpt2_model_configs[cfgs.choose_model])
    base_config = DotDict(base_config)

    # huggingface gpt2 model
    gpt2_hf = GPT2Model.from_pretrained(
        gpt2_huggingface_models[cfgs.choose_model],
        cache_dir = cfgs.pretrained_model_path,
    )
    gpt2_hf.eval()

    # custom gpt model
    model = model_cls(base_config)
    load_weights(model, gpt2_hf, base_config)
    model.eval()

    return model, base_config


def load_pretrained_model(cfgs, model_cls):
    """
    initializing a model with pretrained weights
    """
    # Loading pretrained LLM
    base_config = {
        "vocab_size": cfgs.vocab_size,          # Vocabulary size: 50257
        "context_length": cfgs.context_length,  # Context length: 1024
        "dropout": cfgs.dropout,                # Dropout rate: 0.0
        "qkv_bias": cfgs.qkv_bias,              # Query-key-value bias: True
    }
    base_config.update(gpt2_model_configs[cfgs.choose_model])
    base_config = DotDict(base_config)

    # huggingface gpt2 model
    gpt2_hf = GPT2Model.from_pretrained(
        gpt2_huggingface_models[cfgs.choose_model],
        cache_dir = cfgs.pretrained_model_path,
    )
    gpt2_hf.eval()

    # custom gpt model
    model = model_cls(base_config)
    model.load_state_dict(torch.load(
        cfgs.model_path, 
        map_location=torch.device("cpu"), 
        weights_only=True)
    )
    load_weights(model, gpt2_hf, base_config)
    model.eval()

    return model, base_config




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
