# -*- coding: utf-8 -*-

# ***************************************************
# * File        : load_pretrained_weights.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-01-29
# * Version     : 1.0.012907
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# * TODO        : 1.
# ***************************************************

__all__ = []

# python libraries
import os
import sys
ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import torch

from models.gpt import Model 
from model_train.gpt_generate import generate
from tokenizer.tokenization import token_ids_to_text, text_to_token_ids
from model_load.gtp_download import download_and_load_gpt2
from utils.argsparser_tools import DotDict
from utils.device import device
from utils.log_util import logger

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


def load_weights_into_gpt(gpt, params):
    """
    load the OpenAI weights to correspondiing 
    weight tensors in custom model instance

    Args:
        gpt (_type_): _description_
        params (_type_): _description_
    """
    # ------------------------------
    # assign func
    # ------------------------------
    def _assign(left, right):
        if left.shape != right.shape:
            raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
        return torch.nn.Parameter(torch.tensor(right))
    # ------------------------------
    # assign
    # ------------------------------ 
    # token embedding
    gpt.tok_emb.weight = _assign(gpt.tok_emb.weight, params["wte"])
    # position embedding
    gpt.pos_emb.weight = _assign(gpt.pos_emb.weight, params["wpe"])
    # transformer block
    for b in range(len(params["blocks"])):
        q_w, k_w, v_w = np.split((params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis = -1)
        gpt.trf_blocks[b].attn.W_query.weight = _assign(gpt.trf_blocks[b].attn.W_query.weight, q_w.T)
        gpt.trf_blocks[b].attn.W_key.weight = _assign(gpt.trf_blocks[b].attn.W_key.weight, k_w.T)
        gpt.trf_blocks[b].attn.W_value.weight = _assign(gpt.trf_blocks[b].attn.W_value.weight, v_w.T)

        q_b, k_b, v_b = np.split((params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis = -1)
        gpt.trf_blocks[b].attn.W_query.bias = _assign(gpt.trf_blocks[b].attn.W_query.bias, q_b)
        gpt.trf_blocks[b].attn.W_key.bias = _assign(gpt.trf_blocks[b].attn.W_key.bias, k_b)
        gpt.trf_blocks[b].attn.W_value.bias = _assign(gpt.trf_blocks[b].attn.W_value.bias, v_b)

        gpt.trf_blocks[b].attn.out_proj.weight = _assign(
            gpt.trf_blocks[b].attn.out_proj.weight, 
            params["blocks"][b]["attn"]["c_proj"]["w"].T
        )
        gpt.trf_blocks[b].attn.out_proj.bias = _assign(
            gpt.trf_blocks[b].attn.out_proj.bias, 
            params["blocks"][b]["attn"]["c_proj"]["b"]
        )

        gpt.trf_blocks[b].ff.layers[0].weight = _assign(
            gpt.trf_blocks[b].ff.layers[0].weight, 
            params["blocks"][b]["mlp"]["c_fc"]["w"].T
        )
        gpt.trf_blocks[b].ff.layers[0].bias = _assign(
            gpt.trf_blocks[b].ff.layers[0].bias, 
            params["blocks"][b]["mlp"]["c_fc"]["b"]
        )
        gpt.trf_blocks[b].ff.layers[2].weight = _assign(
            gpt.trf_blocks[b].ff.layers[2].weight, 
            params["blocks"][b]["mlp"]["c_proj"]["w"].T
        )
        gpt.trf_blocks[b].ff.layers[2].bias = _assign(
            gpt.trf_blocks[b].ff.layers[2].bias, 
            params["blocks"][b]["mlp"]["c_proj"]["b"]
        )

        gpt.trf_blocks[b].norm1.scale = _assign(
            gpt.trf_blocks[b].norm1.scale, 
            params["blocks"][b]["ln_1"]["g"]
        )
        gpt.trf_blocks[b].norm1.shift = _assign(
            gpt.trf_blocks[b].norm1.shift, 
            params["blocks"][b]["ln_1"]["b"]
        )
        gpt.trf_blocks[b].norm2.scale = _assign(
            gpt.trf_blocks[b].norm2.scale, 
            params["blocks"][b]["ln_2"]["g"]
        )
        gpt.trf_blocks[b].norm2.shift = _assign(
            gpt.trf_blocks[b].norm2.shift, 
            params["blocks"][b]["ln_2"]["b"]
        )
    # final layer norm
    gpt.final_norm.scale = _assign(gpt.final_norm.scale, params["g"])
    gpt.final_norm.shift = _assign(gpt.final_norm.shift, params["b"])
    # output head linear
    gpt.out_head.weight = _assign(gpt.out_head.weight, params["wte"])


def build_model():
    # pretrained model
    choose_model = "gpt2-small (124M)"
    # ------------------------------
    # model downloading
    # ------------------------------
    model_size = choose_model.split(" ")[-1].lstrip("(").rstrip(")")
    settings, params = download_and_load_gpt2(
        model_size = model_size, 
        models_dir = "./downloaded_models/gpt2_model/"
    )
    logger.info(f"Settings: {settings}")
    logger.info(f"Parameter dictionary keys: {params.keys()}")
    logger.info(f"Token embedding weight tensor(wte): \n{params['wte']}")
    logger.info(f"Token embedding weight tensor dimensions: {params['wte'].shape}")

    # ------------------------------
    # update model config
    # ------------------------------
    # define model config in a dictionary for compactness
    pretrained_model_configs = {
        "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
        "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
        "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
        "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
    }
    # copy the base config and update with speicfic model settings
    GPT_CONFIG_124M = {
        "vocab_size": 50257,
        "context_length": 256,
        "emb_dim": 768,
        "n_heads": 12,
        "n_layers": 12,
        "dropout": 0.1,
        "qkv_bias": False,
    }
    base_config = GPT_CONFIG_124M.copy()
    base_config.update(pretrained_model_configs[choose_model])
    base_config.update({"context_length": 1024, "qkv_bias": True})
    base_config = DotDict(base_config)
    logger.info(f"New config: {base_config}")

    # ------------------------------
    # custom model
    # ------------------------------
    gpt = Model(base_config)
    gpt.eval();

    # ------------------------------
    # load weights
    # ------------------------------
    load_weights_into_gpt(gpt, params)
    gpt.eval()
    
    return gpt, base_config, choose_model




# 测试代码 main 函数
def main():
    # model
    gpt, base_config, choose_model = build_model()
    gpt.to(device)

    # model inference
    torch.manual_seed(123)
    token_ids = generate(
        model=gpt,
        token_idx=text_to_token_ids("Every effort moves you").to(device),
        max_new_tokens=25,
        context_size=base_config.context_length,
        top_k=50,
        temperature=1.5
    )
    logger.info(f"Output text: \n{token_ids_to_text(token_ids)}")

if __name__ == "__main__":
    main()
