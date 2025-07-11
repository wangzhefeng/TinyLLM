# -*- coding: utf-8 -*-

# ***************************************************
# * File        : meta_llama3_weights_load_hf.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-04-04
# * Version     : 1.0.040417
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
from safetensors.torch import load_file

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


def download_llama3_model(model_path: str):
    combined_weights = {}
    # download
    for i in range(1, 5):
        model_name = f"model-0000{i}-of-00004.safetensors"
        model_file_path = Path(model_path).joinpath(model_name)
        if not Path(model_file_path).exists():
            weights_file = hf_hub_download(
                repo_id = "meta-llama/Meta-Llama-3-8B",
                filename = model_name,
                local_dir = "Meta-Llama-3-8B",
            )
        else:
            weights_file = model_file_path
        # load
        weights = load_file(weights_file)
        # logger.info(f"weights keys: {list(weights.keys())}")
        combined_weights.update(weights)

    return combined_weights


def download_llama3_instruct_model(model_path: str): 
    combined_weights = {}
    # download
    for i in range(1, 5):
        model_name = f"model-0000{i}-of-00004.safetensors"
        model_file_path = Path(model_path).joinpath(model_name)
        if not Path(model_file_path).exists():
            weights_file = hf_hub_download(
                repo_id = "meta-llama/Meta-Llama-3-8B",
                filename = model_name,
                local_dir = "Meta-Llama-3-8B-Instruct",
            )
        else:
            weights_file = model_file_path
        # load
        weights = load_file(weights_file)
        # logger.info(f"weights keys: {list(weights.keys())}")
        combined_weights.update(weights)
    
    return combined_weights


def load_weights_into_llama(model, params, param_config):
    # assign function
    def assign(left, right, tensor_name="unknown"):
        if left.shape != right.shape:
            raise ValueError(f"Shape mismatch in tensor '{tensor_name}'. Left: {left.shape}, Right: {right.shape}")

        if isinstance(right, torch.Tensor):
            return torch.nn.Parameter(right.clone().detach())
        else:
            return torch.nn.Parameter(torch.tensor(right))
    
    # embedding
    model.tok_emb.weight = assign(model.tok_emb.weight, params["model.embed_tokens.weight"], "model.embed_tokens.weight")
    # other layers
    for l in range(param_config["n_layers"]):
        # Load attention weights
        model.trf_blocks[l].attn.W_query.weight = assign(
            model.trf_blocks[l].attn.W_query.weight,
            params[f"model.layers.{l}.self_attn.q_proj.weight"],
            f"model.layers.{l}.self_attn.q_proj.weight"
        )
        model.trf_blocks[l].attn.W_key.weight = assign(
            model.trf_blocks[l].attn.W_key.weight,
            params[f"model.layers.{l}.self_attn.k_proj.weight"],
            f"model.layers.{l}.self_attn.k_proj.weight"
        )
        model.trf_blocks[l].attn.W_value.weight = assign(
            model.trf_blocks[l].attn.W_value.weight,
            params[f"model.layers.{l}.self_attn.v_proj.weight"],
            f"model.layers.{l}.self_attn.v_proj.weight"
        )
        model.trf_blocks[l].attn.out_proj.weight = assign(
            model.trf_blocks[l].attn.out_proj.weight,
            params[f"model.layers.{l}.self_attn.o_proj.weight"],
            f"model.layers.{l}.self_attn.o_proj.weight"
        )
        model.trf_blocks[l].norm1.weight = assign(
            model.trf_blocks[l].norm1.weight,
            params[f"model.layers.{l}.input_layernorm.weight"],
            f"model.layers.{l}.input_layernorm.weight"
        )

        # Load FeedForward weights
        model.trf_blocks[l].ff.fc1.weight = assign(
            model.trf_blocks[l].ff.fc1.weight,
            params[f"model.layers.{l}.mlp.gate_proj.weight"],
            f"model.layers.{l}.mlp.gate_proj.weight"
        )
        model.trf_blocks[l].ff.fc2.weight = assign(
            model.trf_blocks[l].ff.fc2.weight,
            params[f"model.layers.{l}.mlp.up_proj.weight"],
            f"model.layers.{l}.mlp.up_proj.weight"
        )
        model.trf_blocks[l].ff.fc3.weight = assign(
            model.trf_blocks[l].ff.fc3.weight,
            params[f"model.layers.{l}.mlp.down_proj.weight"],
            f"model.layers.{l}.mlp.down_proj.weight"
        )
        model.trf_blocks[l].norm2.weight = assign(
            model.trf_blocks[l].norm2.weight,
            params[f"model.layers.{l}.post_attention_layernorm.weight"],
            f"model.layers.{l}.post_attention_layernorm.weight"
        )

    # Load output layer weights
    model.final_norm.weight = assign(model.final_norm.weight, params["model.norm.weight"], "model.norm.weight")

    if "lm_head.weight" in params.keys():
        model.out_head.weight = assign(model.out_head.weight, params["lm_head.weight"], "lm_head.weight")
    else:
        model.out_head.weight = assign(model.out_head.weight, params["model.embed_tokens.weight"], "model.embed_tokens.weight")
        print("Model uses weight tying")

    return model




# 测试代码 main 函数
def main():
    # Meta/Llama-2-7b model weights load
    model_path = "downloaded_models/llama_model/Meta-Llama-3-8B"
    weights = download_llama3_model(model_path) 
 
    chat_model_path = "downloaded_models/llama_model/Meta-Llama-3-8B-Instruct"
    weights_instruct = download_llama3_instruct_model(model_path)

if __name__ == "__main__":
    main()
