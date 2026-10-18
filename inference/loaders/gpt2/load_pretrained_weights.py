# -*- coding: utf-8 -*-

# ***************************************************
# * File        : load_pretrained_weights.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-03-02
# * Version     : 0.1.030223
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)

import torch
import torch.nn as nn

from utils.args_tools import DotDict

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


def load_pretrained_model(cfgs, model_configs, model_cls, device: str = "cpu", task: str = "binary_classification"):
    """
    initializing a model with pretrained weights
    """
    # huggingface gpt2 pretrained model
    # gpt2_hf = GPT2Model.from_pretrained(
    #     gpt2_huggingface_models[cfgs.choose_model],
    #     cache_dir = cfgs.pretrained_model_path,
    # )
    # gpt2_hf.eval()

    # update pretrained model's config
    base_config = {
        "vocab_size": cfgs.vocab_size,          # Vocabulary size: 50257
        "context_length": cfgs.context_length,  # Context length: 1024
        "dropout": cfgs.dropout,                # Dropout rate: 0.0
        "qkv_bias": cfgs.qkv_bias,              # Query-key-value bias: True
    }
    base_config.update(model_configs[cfgs.choose_model])
    base_config = DotDict(base_config)
    
    # pretrained model instance
    model = model_cls(base_config)
    if task == "binary_classification":
        model.out_head = nn.Linear(
            in_features=cfgs.emb_dim, 
            out_features=cfgs.num_classes
        )
    model.load_state_dict(torch.load(
        cfgs.finetuned_model_path, 
        map_location = device, 
        weights_only = True
    ))
    model.to(device)
    # assign pretrained model's weights
    # load_weights_hf(model, gpt2_hf, base_config)

    # model inference mode
    model.eval()

    return model




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
