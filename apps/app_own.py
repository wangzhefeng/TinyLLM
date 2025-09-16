# -*- coding: utf-8 -*-

# ***************************************************
# * File        : app_orig.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-09-07
# * Version     : 1.0.090713
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
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import torch
import chainlit

from layers.tokenizers.tokenization import (
    choose_tokenizer,
    text_to_token_ids,
    token_ids_to_text,
)
from models.gpt2_124M import Model
from layers.inference import generate
from utils.args_tools import DotDict
from utils.device import device_setting

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]
os.environ['LOG_NAME'] = LOGGING_LABEL
from utils.log_util import logger


# device
device = device_setting(verbose=True)


def get_model_and_tokenizer():
    # model config
    model_config = {
        "vocab_size": 50257,     # Vocabular size
        "context_length": 256,  # Context length
        "max_new_toknes": 50,   # Maximum new tokens to generate
        "embed_dim": 768,        # Embedding dimension
        "d_ff": 4 * 768,         # Hidden dimension
        "n_heads": 12,           # Number of attention heads
        "n_layers": 12,          # Number of transformer layers
        "dropout": 0.1,          # Dropout rate
        "qkv_bias": False,       # Query-Key-Value bias
        "dtype": torch.float32,
        "kv_window_size": 1024,  # KV cache window size
        "top_k": 1,
        "temperature": 0.0,
        "eos_id": 50256,
        "use_cache": True,
        "tokenizer_model": "tiktoken_gpt2_bpe",
    }
    model_config = DotDict(model_config)
    # tokenizer
    tokenizer = choose_tokenizer(tokenizer_model = "tiktoken_gpt2_bpe")
    # model
    model_path = Path("./model.pth")
    if not model_path.exists():
        logger.info(f"Could not find the {model_path} file.")
        sys.exit()
    checkpoint = torch.load(model_path, weights_only=True)
    model = Model(model_config)
    model.load_state_dict(checkpoint)
    model.to(device)

    return tokenizer, model, model_config


# Obtain the necessary tokenizer and model files for the chainlit function below
tokenizer, model, model_config = get_model_and_tokenizer()




# 测试代码 main 函数
@chainlit.on_message
async def main(message: chainlit.Message):
    """
    The Main Chainlit function
    """
    token_ids = generate(
        model=model,
        token_idx=text_to_token_ids(
            message.content, 
            tokenizer_model=model_config.tokenizer_model
        ).to(device),
        max_new_tokens=model_config.max_new_toknes,
        context_length=model_config.context_length,
        temperature=model_config.temperature,
        top_k=model_config.top_k,
        eos_id=model_config.eos_id,
        use_cache=model_config.use_cache,
    )
    text = tokenizer.decode(token_ids, tokenizer)

    # Returns the model response to the interface
    await chainlit.Message(content=f"{text}").send()

if __name__ == "__main__":
    main()

