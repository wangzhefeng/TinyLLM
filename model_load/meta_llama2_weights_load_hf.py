# -*- coding: utf-8 -*-

# ***************************************************
# * File        : meta_llama2_weights_load.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-03-30
# * Version     : 1.0.033017
# * Description : description
# * Link        : link
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


import torch
from huggingface_hub import hf_hub_download

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


def download_llama2_model(model_path: str):
    model_file_path = Path(model_path).joinpath("consolidated.00.pth")
    # download
    if not Path(model_file_path).exists():
        weights_file = hf_hub_download(
            repo_id = "meta-llama/Llama-2-7b",
            filename = "consolidated.00.pth",
            local_dir = "Llama-2-7b",
        )
    else:
        weights_file = model_file_path
    # load
    weights = torch.load(weights_file, weights_only = True)
    # logger.info(f"weights keys: {list(weights.keys())}")

    return weights


def download_llama2_chat_model(model_path: str):
    model_file_path = Path(model_path).joinpath("consolidated.00.pth")
    # download
    if not Path(model_path).exists():
        weights_file = hf_hub_download(
            repo_id = "meta-llama/Llama-2-7b-chat",
            filename = "consolidated.00.pth",
            local_dir = "Llama-2-7b-chat",
        )
    else:
        weights_file = model_path
    # load
    weights = torch.load(weights_file, weights_only = True)
    # logger.info(f"weights keys: {list(weights.keys())}")

    return weights


def load_weights_into_llama(model, params, param_config):
    # assign function
    def assign(left, right):
        if left.shape != right.shape:
            raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")

        if isinstance(right, torch.Tensor):
            return torch.nn.Parameter(right.clone().detach())
        else:
            return torch.nn.Parameter(torch.tensor(right))
    
    # embedding
    model.tok_emb.weight = assign(model.tok_emb.weight, params["tok_embeddings.weight"])
    # other layers
    for l in range(param_config["n_layers"]):
        # Load attention weights
        model.trf_blocks[l].attn.W_query.weight = assign(
            model.trf_blocks[l].attn.W_query.weight,
            params[f"layers.{l}.attention.wq.weight"]
        )
        model.trf_blocks[l].attn.W_key.weight = assign(
            model.trf_blocks[l].attn.W_key.weight,
            params[f"layers.{l}.attention.wk.weight"]
        )
        model.trf_blocks[l].attn.W_value.weight = assign(
            model.trf_blocks[l].attn.W_value.weight,
            params[f"layers.{l}.attention.wv.weight"]
        )
        model.trf_blocks[l].attn.out_proj.weight = assign(
            model.trf_blocks[l].attn.out_proj.weight,
            params[f"layers.{l}.attention.wo.weight"]
        )
        model.trf_blocks[l].norm1.weight = assign(
            model.trf_blocks[l].norm1.weight,
            params[f"layers.{l}.attention_norm.weight"]
        )

        # Load FeedForward weights
        model.trf_blocks[l].ff.fc1.weight = assign(
            model.trf_blocks[l].ff.fc1.weight,
            params[f"layers.{l}.feed_forward.w1.weight"]
        )
        # For some reason w2 and w3 are provided in the wrong order in the weights file
        model.trf_blocks[l].ff.fc2.weight = assign(
            model.trf_blocks[l].ff.fc2.weight,
            params[f"layers.{l}.feed_forward.w3.weight"]
        )
        model.trf_blocks[l].ff.fc3.weight = assign(
            model.trf_blocks[l].ff.fc3.weight,
            params[f"layers.{l}.feed_forward.w2.weight"]
        )
        model.trf_blocks[l].norm2.weight = assign(
            model.trf_blocks[l].norm2.weight,
            params[f"layers.{l}.ffn_norm.weight"]
        )

    # Load output layer weights
    model.final_norm.weight = assign(model.final_norm.weight, params["norm.weight"])
    model.out_head.weight = assign(model.out_head.weight, params["output.weight"])
    
    return model




# 测试代码 main 函数
def main():
    # Meta/Llama-2-7b model weights load
    model_path = "downloaded_models/llama_model/Llama-2-7b"
    weights = download_llama2_model(model_path) 
 
    chat_model_path = "downloaded_models/llama_model/Llama-2-7b-chat"
    weights_chat = download_llama2_chat_model(model_path)

if __name__ == "__main__":
    main()
