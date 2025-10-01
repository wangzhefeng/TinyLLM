# -*- coding: utf-8 -*-

# ***************************************************
# * File        : qwen3_06B_pretrained.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-09-29
# * Version     : 1.0.092923
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
import warnings
warnings.filterwarnings("ignore")

import torch

from utils.llm.reasoning_from_scratch.qwen3 import (
    download_qwen3_small,
    Qwen3Model,
    QWEN_CONFIG_06_B,
)
from utils.device import device_setting
from layers.tokenizers.qwen3_tokenizer import tokenizer

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]
os.environ['LOG_NAME'] = LOGGING_LABEL
from utils.log_util import logger


# device
device = device_setting(verbose=True)
# device = torch.device("cpu")

# llm model path
llm_model_dir = "./downloaded_models/qwen3_model"
llm_model_path = Path(llm_model_dir).joinpath("qwen3-0.6B-base.pth")

# llm model download
if not llm_model_path.exists():
    download_qwen3_small(
        kind="base", 
        tokenizer_only=False, 
        out_dir=llm_model_dir,
    )

# llm model
model = Qwen3Model(QWEN_CONFIG_06_B)
model.load_state_dict(torch.load(llm_model_path))
model.to(device)
# logger.info(f"model: \n{model}")

# model compile
major, minor = map(int, torch.__version__.split(".")[:2])
if torch.cuda.is_available():
    # if (major, minor) > (2, 8):
    #     # This avoids retriggering model recompilations in PyTorch 2.8 and newer
    #     # if the model contains code like self.pos = self.pos + 1
    #     torch._dynamo.config.allow_unspec_int_on_nn_module = True
    model_compiled = torch.compile(model)




# 测试代码 main 函数
def main():
    import time

    from utils.args_tools import DotDict
    from layers.inference import generate, generate_qwen3, generate_stats


    # input
    prompt = "Explain large language models in a single sentence."
    input_token_ids_tensor = torch.tensor(tokenizer.encode(prompt), device=device).unsqueeze(0)

    """
    # inference 1
    output_token_ids_tensor = generate(
        model = model,
        token_idx = input_token_ids_tensor,
        max_new_tokens = 100,
        context_length=DotDict(QWEN_CONFIG_06_B).context_length,
    )
    output_text = tokenizer.decode(output_token_ids_tensor.squeeze(0).tolist())
    logger.info(f"output_text: \n{output_text}")

    # inference 2
    output_token_ids_tensor = generate(
        model = model,
        token_idx = input_token_ids_tensor,
        max_new_tokens = 100,
        eos_token_id = tokenizer.eos_token_id,
        context_length=DotDict(QWEN_CONFIG_06_B).context_length,
    )
    output_text = tokenizer.decode(output_token_ids_tensor.squeeze(0).tolist())
    logger.info(f"output_text: \n{output_text}")
    """
    # inference 3
    start_time = time.time()
    output_token_ids_tensor = generate_qwen3(
        model = model,
        token_idx = input_token_ids_tensor,
        max_new_tokens = 100,
        eos_token_id = tokenizer.eos_token_id,
        context_length=DotDict(QWEN_CONFIG_06_B).context_length,
        use_cache = False,
    )
    output_text = tokenizer.decode(output_token_ids_tensor.squeeze(0).tolist())
    logger.info(f"output_text: \n{output_text}")
    end_time = time.time()
    generate_stats(
        output_token_ids = output_token_ids_tensor,
        tokenizer = tokenizer,
        start_time = start_time,
        end_time = end_time,
    )

    # inference 4
    start_time = time.time()
    output_token_ids_tensor = generate_qwen3(
        model = model,
        token_idx = input_token_ids_tensor,
        max_new_tokens = 100,
        context_length = DotDict(QWEN_CONFIG_06_B).context_length,
        eos_token_id = tokenizer.eos_token_id,
        use_cache = True,
    )
    output_text = tokenizer.decode(output_token_ids_tensor.squeeze(0).tolist())
    logger.info(f"output_text: \n{output_text}")
    end_time = time.time()
    generate_stats(
        output_token_ids = output_token_ids_tensor,
        tokenizer = tokenizer,
        start_time = start_time,
        end_time = end_time,
    )

    # inference 5
    start_time = time.time()
    output_token_ids_tensor = generate_qwen3(
        model = model_compiled,
        token_idx = input_token_ids_tensor,
        max_new_tokens = 100,
        context_length = DotDict(QWEN_CONFIG_06_B).context_length,
        eos_token_id = tokenizer.eos_token_id,
        use_cache = True,
    )
    output_text = tokenizer.decode(output_token_ids_tensor.squeeze(0).tolist())
    logger.info(f"output_text: \n{output_text}")
    end_time = time.time()
    generate_stats(
        output_token_ids = output_token_ids_tensor,
        tokenizer = tokenizer,
        start_time = start_time,
        end_time = end_time,
    )

if __name__ == "__main__":
    main()
