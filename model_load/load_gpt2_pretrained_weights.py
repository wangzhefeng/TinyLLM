# -*- coding: utf-8 -*-

# ***************************************************
# * File        : opeai_gpt2_models.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-03-02
# * Version     : 0.1.030223
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


from transformers import GPT2Model

from model_load.openai_gpt2_weights_load_hf import load_weights_hf
from model_load.openai_gpt2_weights_load_hf_safetensors import (
    download_and_load_gpt2_st, 
    load_weights_hf_safetensors,
)
from model_load.openai_gpt2_weights_load import (
    download_and_load_gpt2, 
    load_weights_download,
)
from utils.args_tools import DotDict
from utils.log_util import logger

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


# huggingface allowed model names
gpt2_model_names = {
    "gpt2-small (124M)": "gpt2",         # works ok
    "gpt2-medium (355M)": "gpt2-medium", # this file seems to have issues via `generate`
    "gpt2-large (774M)": "gpt2-large",   # works ok
    "gpt2-xl (1558M)": "gpt2-xl"         # works ok
}
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


def model_with_gpt2_weights(cfgs, model_cls, model_source: str = "huggingface_gpt2"):
    """
    initializing a model with pretrained weights
    """
    # Loading pretrained LLM
    base_config = {
        "vocab_size": cfgs.vocab_size,          # Vocabulary size: 50257
        "context_length": cfgs.context_length,  # Context length: 1024
        "dropout": cfgs.dropout,                # Dropout rate: 0.0
        "qkv_bias": cfgs.qkv_bias,              # Query-key-value bias: True
        "emb_dim": cfgs.emb_dim,                # Embedding dimension: 768
        "n_layers": cfgs.n_layers,              # Number of layers: 12
        "n_heads": cfgs.n_heads,                # Number of heads: 12
    }
    base_config.update(gpt2_model_configs[cfgs.pretrained_model])
    base_config = DotDict(base_config)
    logger.info(f"New config: {base_config}")

    # pretrained LLM model
    model = model_cls(base_config)
    model.eval()

    # load pretrained model's weights
    if model_source == "huggingface_gpt2":
        # huggingface gpt2 model
        gpt2_hf = GPT2Model.from_pretrained(
            gpt2_huggingface_models[cfgs.pretrained_model],
            cache_dir = cfgs.pretrained_model_path,
        )
        gpt2_hf.eval()
        # load weights of gpt2 model to pretrained LLM model
        load_weights_hf(model, gpt2_hf, base_config) 
    elif model_source == "huggingface_gpt2_safetensors":
        state_dict = download_and_load_gpt2_st(
            gpt2_model_names, 
            cfgs.pretrained_model
        )
        load_weights_hf_safetensors(model, state_dict)
    elif model_source == "download_and_load_gpt2":
        model_size = cfgs.pretrained_model.split(" ")[-1].lstrip("(").rstrip(")")
        settings, params = download_and_load_gpt2(
            model_size = model_size, 
            models_dir = cfgs.pretrained_model_path,
        )
        logger.info(f"Settings: {settings}")
        logger.info(f"Parameter dictionary keys: {params.keys()}")
        logger.info(f"Token embedding weight tensor(wte): \n{params['wte']}")
        logger.info(f"Token embedding weight tensor dimensions: {params['wte'].shape}")
        # load weights of gpt2 model to pretrained LLM model
        load_weights_download(model, params)

    # model inference mode
    model.eval()

    return model, base_config




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
