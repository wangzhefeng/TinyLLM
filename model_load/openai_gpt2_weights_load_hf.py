# -*- coding: utf-8 -*-

# ***************************************************
# * File        : openai_gpt2_pretrained_weight_load_hf.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-02-15
# * Version     : 0.1.021518
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

import numpy as np
import torch

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


def load_weights_hf(gpt, gpt_hf, CONFIG):
    # assign function
    def assign_check(left, right):
        if left.shape != right.shape:
            raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
        return torch.nn.Parameter(right.clone().detach())
    
    # gpt weights
    d = gpt_hf.state_dict()

    # embedding
    gpt.pos_emb.weight = assign_check(gpt.pos_emb.weight, d["wpe.weight"])
    gpt.tok_emb.weight = assign_check(gpt.tok_emb.weight, d["wte.weight"])
    # other layers
    for b in range(CONFIG["n_layers"]):
        q_w, k_w, v_w = np.split(d[f"h.{b}.attn.c_attn.weight"], 3, axis=-1)
        gpt.trf_blocks[b].attn.W_query.weight = assign_check(gpt.trf_blocks[b].attn.W_query.weight, q_w.T)
        gpt.trf_blocks[b].attn.W_key.weight = assign_check(gpt.trf_blocks[b].attn.W_key.weight, k_w.T)
        gpt.trf_blocks[b].attn.W_value.weight = assign_check(gpt.trf_blocks[b].attn.W_value.weight, v_w.T)
    
        q_b, k_b, v_b = np.split(d[f"h.{b}.attn.c_attn.bias"], 3, axis=-1)
        gpt.trf_blocks[b].attn.W_query.bias = assign_check(gpt.trf_blocks[b].attn.W_query.bias, q_b)
        gpt.trf_blocks[b].attn.W_key.bias = assign_check(gpt.trf_blocks[b].attn.W_key.bias, k_b)
        gpt.trf_blocks[b].attn.W_value.bias = assign_check(gpt.trf_blocks[b].attn.W_value.bias, v_b)
    
        gpt.trf_blocks[b].attn.out_proj.weight = assign_check(gpt.trf_blocks[b].attn.out_proj.weight, d[f"h.{b}.attn.c_proj.weight"].T)
        gpt.trf_blocks[b].attn.out_proj.bias = assign_check(gpt.trf_blocks[b].attn.out_proj.bias, d[f"h.{b}.attn.c_proj.bias"])
    
        gpt.trf_blocks[b].ff.layers[0].weight = assign_check(gpt.trf_blocks[b].ff.layers[0].weight, d[f"h.{b}.mlp.c_fc.weight"].T)
        gpt.trf_blocks[b].ff.layers[0].bias = assign_check(gpt.trf_blocks[b].ff.layers[0].bias, d[f"h.{b}.mlp.c_fc.bias"])
        gpt.trf_blocks[b].ff.layers[2].weight = assign_check(gpt.trf_blocks[b].ff.layers[2].weight, d[f"h.{b}.mlp.c_proj.weight"].T)
        gpt.trf_blocks[b].ff.layers[2].bias = assign_check(gpt.trf_blocks[b].ff.layers[2].bias, d[f"h.{b}.mlp.c_proj.bias"])
    
        gpt.trf_blocks[b].norm1.scale = assign_check(gpt.trf_blocks[b].norm1.scale, d[f"h.{b}.ln_1.weight"])
        gpt.trf_blocks[b].norm1.shift = assign_check(gpt.trf_blocks[b].norm1.shift, d[f"h.{b}.ln_1.bias"])
        gpt.trf_blocks[b].norm2.scale = assign_check(gpt.trf_blocks[b].norm2.scale, d[f"h.{b}.ln_2.weight"])
        gpt.trf_blocks[b].norm2.shift = assign_check(gpt.trf_blocks[b].norm2.shift, d[f"h.{b}.ln_2.bias"])
    
        gpt.final_norm.scale = assign_check(gpt.final_norm.scale, d[f"ln_f.weight"])
        gpt.final_norm.shift = assign_check(gpt.final_norm.shift, d[f"ln_f.bias"])
        gpt.out_head.weight = assign_check(gpt.out_head.weight, d["wte.weight"])

    return gpt




# 测试代码 main 函数
def main(): 
    from transformers import GPT2Model

    from models.gpt import Model
    from utils.llm.gpt_generate import generate
    from layers.tokenizers.tokenization import text_to_token_ids, token_ids_to_text
    from utils.device import device_setting
    from utils.args_tools import DotDict
    from utils.log_util import logger 

    # huggingface gpt2 model
    choose_model = "gpt2-small (124M)"
    # huggingface allowed model names 
    gpt_hf = GPT2Model.from_pretrained(
        gpt2_model_names[choose_model], 
        cache_dir="./downloaded_models/gpt2_model"
    )
    gpt_hf.eval();

    # custom model config 
    base_config = {
        "vocab_size": 50257,     # Vocabulary size
        "context_length": 1024,  # Context length
        "dropout": 0.0,          # Dropout rate
        "qkv_bias": True         # Query-key-value bias
    }
    base_config.update(gpt2_model_configs[choose_model])
    base_config = DotDict(base_config)

    # device
    device = device_setting()

    # custom model
    gpt = Model(base_config)

    # update weights
    gpt = load_weights_hf(gpt, gpt_hf, base_config) 
    gpt.to(device)

    # model inference
    token_ids = generate(
        model=gpt,
        token_idx=text_to_token_ids("Every effort moves").to(device),
        max_new_tokens=30,
        context_size=base_config.context_length,
        temperature=1.0,
        top_k=1,
        eos_id=50256,
    )
    logger.info(f"Output text: \n{token_ids_to_text(token_ids)}")

if __name__ == "__main__":
    main()
