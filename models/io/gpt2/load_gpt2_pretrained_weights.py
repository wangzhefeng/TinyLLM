# -*- coding: utf-8 -*-

# ***************************************************
# * File        : load_gpt2_pretrained_weights.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-03-02
# * Version     : 0.1.030223
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
from pathlib import Path

from transformers import GPT2Model

import torch

from config.gpt2.model_cfgs import (
    gpt2_model_names,
    gpt2_model_configs,
    gpt2_huggingface_models,
)
from models.io.gpt2.openai_gpt2_weights_load_hf import load_weights_hf
from models.io.gpt2.openai_gpt2_weights_load_hf_safetensors import (
    download_and_load_gpt2_st,
    load_weights_hf_safetensors,
)
from models.io.gpt2.openai_gpt2_weights_load import (
    download_and_load_gpt2,
    load_weights_download,
)
from utils.args_tools import DotDict
from utils.log_util import logger

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


def model_with_gpt2_weights(cfgs, model_cls, model_source: str = "huggingface_gpt2"):
    """
    initializing a model with pretrained weights

    Args:
        cfgs: 需包含 vocab_size/context_length/dropout/qkv_bias/n_heads 与
            embed_dim 或 emb_dim（两者任一；run 脚本多用 emb_dim）、
            pretrained_model（形如 "gpt2-small (124M)"，见 config/gpt2/model_cfgs.py）
    """
    model_source = {
        "hf": "huggingface_gpt2",
        "safetensors": "huggingface_gpt2_safetensors",
        "openai_tf": "download_and_load_gpt2",
    }.get(model_source, model_source)
    # Loading pretrained LLM
    # （embed_dim 兼容 emb_dim：不同 run 脚本参数命名不一）
    embed_dim = getattr(cfgs, "embed_dim", None) or getattr(cfgs, "emb_dim")
    base_config = {
        "vocab_size": cfgs.vocab_size,          # Vocabulary size: 50257
        "context_length": cfgs.context_length,  # Context length: 1024
        "dropout": cfgs.dropout,                # Dropout rate: 0.0
        "qkv_bias": cfgs.qkv_bias,              # Query-key-value bias: True
        "embed_dim": embed_dim,                 # Embedding dimension: 768
        "n_layers": cfgs.n_layers,              # Number of layers: 12
        "n_heads": cfgs.n_heads,                # Number of heads: 12
        "dtype": getattr(cfgs, "dtype", None) or torch.float32,  # layers/feed_forward.py 需要
        "tie_word_embeddings": getattr(cfgs, "tie_word_embeddings", False),  # weight tying 开关
    }
    base_config.update(gpt2_model_configs[cfgs.pretrained_model])
    # GPT-2 FFN 维度 = 4 × embed_dim（需在 config 表覆盖后计算，换 medium/large 才对）
    base_config["d_ff"] = 4 * base_config["embed_dim"]
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
    # weight tying：加载过程可能替换掉共享的 Parameter，此处重新建立共享
    if base_config.tie_word_embeddings:
        model.out_head.weight = model.tok_embed.weight

    return model, base_config




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
