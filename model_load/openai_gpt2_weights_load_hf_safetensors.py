# -*- coding: utf-8 -*-

# ***************************************************
# * File        : openai_gpt2_weights_load_hf_safetensors.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-02-15
# * Version     : 0.1.021519
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
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import urllib.request


import torch
from safetensors.torch import load_file

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


def load_weights_hf_safetensors(gpt, params):
    # assign function
    def assign(left, right):
        if left.shape != right.shape:
            raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
        return torch.nn.Parameter(right.detach())
    
    # embedding
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params["wpe.weight"])
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params["wte.weight"])
    # other layers
    for b in range(len(gpt.trf_blocks)):
        q_w, k_w, v_w = torch.chunk(params[f"h.{b}.attn.c_attn.weight"], 3, axis=-1)
        gpt.trf_blocks[b].attn.W_query.weight = assign(
            gpt.trf_blocks[b].attn.W_query.weight, 
            q_w.T
        )
        gpt.trf_blocks[b].attn.W_key.weight = assign(
            gpt.trf_blocks[b].attn.W_key.weight, 
            k_w.T
        )
        gpt.trf_blocks[b].attn.W_value.weight = assign(
            gpt.trf_blocks[b].attn.W_value.weight, 
            v_w.T
        )

        q_b, k_b, v_b = torch.chunk(params[f"h.{b}.attn.c_attn.bias"], 3, axis=-1)
        gpt.trf_blocks[b].attn.W_query.bias = assign(
            gpt.trf_blocks[b].attn.W_query.bias, 
            q_b
        )
        gpt.trf_blocks[b].attn.W_key.bias = assign(
            gpt.trf_blocks[b].attn.W_key.bias, 
            k_b
        )
        gpt.trf_blocks[b].attn.W_value.bias = assign(
            gpt.trf_blocks[b].attn.W_value.bias, 
            v_b
        )

        gpt.trf_blocks[b].attn.out_proj.weight = assign(
            gpt.trf_blocks[b].attn.out_proj.weight,
            params[f"h.{b}.attn.c_proj.weight"].T
        )
        gpt.trf_blocks[b].attn.out_proj.bias = assign(
            gpt.trf_blocks[b].attn.out_proj.bias,
            params[f"h.{b}.attn.c_proj.bias"]
        )

        gpt.trf_blocks[b].ff.layers[0].weight = assign(
            gpt.trf_blocks[b].ff.layers[0].weight,
            params[f"h.{b}.mlp.c_fc.weight"].T
        )
        gpt.trf_blocks[b].ff.layers[0].bias = assign(
            gpt.trf_blocks[b].ff.layers[0].bias,
            params[f"h.{b}.mlp.c_fc.bias"]
        )
        gpt.trf_blocks[b].ff.layers[2].weight = assign(
            gpt.trf_blocks[b].ff.layers[2].weight,
            params[f"h.{b}.mlp.c_proj.weight"].T
        )
        gpt.trf_blocks[b].ff.layers[2].bias = assign(
            gpt.trf_blocks[b].ff.layers[2].bias,
            params[f"h.{b}.mlp.c_proj.bias"]
        )

        gpt.trf_blocks[b].norm1.scale = assign(
            gpt.trf_blocks[b].norm1.scale,
            params[f"h.{b}.ln_1.weight"]
        )
        gpt.trf_blocks[b].norm1.shift = assign(
            gpt.trf_blocks[b].norm1.shift,
            params[f"h.{b}.ln_1.bias"]
        )
        gpt.trf_blocks[b].norm2.scale = assign(
            gpt.trf_blocks[b].norm2.scale,
            params[f"h.{b}.ln_2.weight"]
        )
        gpt.trf_blocks[b].norm2.shift = assign(
            gpt.trf_blocks[b].norm2.shift,
            params[f"h.{b}.ln_2.bias"]
        )

    # Load output layer weights
    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["ln_f.weight"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["ln_f.bias"])
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte.weight"])

    return gpt


def download_and_load_gpt2_st(gpt_model_names, pretrained_model):
    # params
    url = f"https://huggingface.co/openai-community/{gpt_model_names[pretrained_model]}/resolve/main/model.safetensors"
    output_file = f"downloaded_models/gpt2_model/model-{gpt_model_names[pretrained_model]}.safetensors"
    # download
    if not Path(output_file).exists():
        urllib.request.urlretrieve(url, output_file)
    # load
    state_dict = load_file(output_file)

    return state_dict




# 测试代码 main 函数
def main():
    from models.gpt import Model
    from utils.llm.gpt_generate import generate
    from layers.tokenizers.tokenization import text_to_token_ids, token_ids_to_text
    from utils.device import device_setting
    from utils.args_tools import DotDict
    from utils.log_util import logger

    # device
    device = device_setting()

    # huggingface allowed model names
    model_names = {
        "gpt2-small (124M)": "gpt2",         # works ok
        "gpt2-medium (355M)": "gpt2-medium", # this file seems to have issues via `generate`
        "gpt2-large (774M)": "gpt2-large",   # works ok
        "gpt2-xl (1558M)": "gpt2-xl"         # works ok
    }
    model_configs = {
        "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
        "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
        "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
        "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
    }

    # huggingface gpt2 model
    choose_model = "gpt2-small (124M)"
    state_dict = download_and_load_gpt2_st(model_names, choose_model)

    # custom model config
    base_config = {
        "vocab_size": 50257,    # Vocabulary size
        "context_length": 1024, # Context length
        "dropout": 0.0,       # Dropout rate
        "qkv_bias": True        # Query-key-value bias
    }
    base_config.update(model_configs[choose_model])
    base_config = DotDict(base_config)
    # custom model
    gpt = Model(base_config)

    # update weights
    load_weights_hf_safetensors(gpt, state_dict)
    gpt.to(device)

    # model inference
    torch.manual_seed(123)
    token_ids = generate(
        model=gpt,
        token_idx=text_to_token_ids("Every effort moves").to(device),
        max_new_tokens=30,
        context_size=base_config.context_length,
        top_k=1,
        temperature=1.0,
        eos_id=50256,
    )
    logger.info(f"Output text: \n{token_ids_to_text(token_ids)}")

if __name__ == "__main__":
    main()
