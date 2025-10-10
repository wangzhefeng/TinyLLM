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

from config.qwen3.model_cfgs import get_cfgs
from config.qwen3.model_download import download_from_huggingface
from models.qwen3.qwen3_original import Model
from layers.tokenizers.qwen3.qwen3_tokenizer import Qwen3Tokenizer
from layers.inference import generate_qwen3_stream
from utils.device import device_setting

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]
os.environ['LOG_NAME'] = LOGGING_LABEL
from utils.log_util import logger


# ------------------------------
# setting
# ------------------------------
# lower precision from "highest"(default) which enables Tensor Cores if applicable
torch.set_float32_matmul_precision("high")

# device
device = device_setting(verbose=True)
# device = torch.device("cpu")

# ------------------------------
# llm model and tokenizer download
# ------------------------------
# model type
model_type = "base"

# llm model path
model_dir = Path(f"./downloaded_models/qwen3_model/Qwen3-0.6B-{model_type}-for-reasoning")

# llm model download
llm_model_path, tokenizer_path = download_from_huggingface(
    kind=model_type, 
    tokenizer_only=False, 
    repo_id="rasbt/qwen3-from-scratch",
    revision="main",
    local_dir=model_dir,
)

# ------------------------------
# llm model and tokenizer
# ------------------------------
# llm model config
QWEN3_CONFIG = get_cfgs(CHOOSE_MODEL="0.6B")
# llm model
model = Model(QWEN3_CONFIG)
# load model weights
model.load_state_dict(torch.load(llm_model_path))
# move model to device
model.to(device)
# model compile
major, minor = map(int, torch.__version__.split(".")[:2])
if torch.cuda.is_available():
    if (major, minor) > (2, 8):
        # This avoids retriggering model recompilations in PyTorch 2.8 and newer
        # if the model contains code like self.pos = self.pos + 1
        torch._dynamo.config.allow_unspec_int_on_nn_module = True
    model_compiled = torch.compile(model)

# tokenizer
tokenizer = Qwen3Tokenizer(tokenizer_file_path=tokenizer_path)

# ------------------------------
# model inference
# ------------------------------
def generate_qwen3_stream_concat(model, tokenizer, prompt, device, max_new_tokens=521, verbose=False, display=False):
    # input prompt token ids
    input_ids = torch.tensor(tokenizer.encode(prompt), device=device).unsqueeze(0)
    # inference
    generated_ids = []
    for token in generate_qwen3_stream(
        model=model,
        token_ids=input_ids,
        max_new_tokens=max_new_tokens,
        context_length=QWEN3_CONFIG.context_length,
        temperature=0.0,
        top_k=None,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
    ):
        next_token_id = token.squeeze(0)
        generated_ids.append(next_token_id.item())
        if verbose:
            print(tokenizer.decode(next_token_id.tolist()), end="", flush=True)
    # inference result
    generated_tokens = tokenizer.decode(generated_ids)
    if display:
        from IPython.display import Latex, Math, display
        display(Latex(generated_tokens))
        # display(Math(r"\dfrac{14}{3}"))
    
    return generated_tokens


# inference_res = generate_qwen3_stream_concat(
#     model=model, 
#     tokenizer=tokenizer, 
#     prompt=(
#         r"If $a+b=3$ and $ab=\tfrac{13}{6}$, "
#         r"what is the value of $a^2+b^2$?"
#     ), 
#     device=device, 
#     max_new_tokens=521, 
#     verbose=True, 
#     display=False,
# )
# logger.info(inference_res)




# 测试代码 main 函数
def main():
    """
    import time

    from utils.args_tools import DotDict
    from layers.inference import generate, generate_qwen3, generate_stats


    # input
    prompt = "Explain large language models in a single sentence."
    input_token_ids_tensor = torch.tensor(tokenizer.encode(prompt), device=device).unsqueeze(0)

    # inference 1
    # -----------
    output_token_ids_tensor = generate(
        model = model,
        token_ids = input_token_ids_tensor,
        max_new_tokens = 100,
        context_length=DotDict(QWEN_CONFIG_06_B).context_length,
    )
    output_text = tokenizer.decode(output_token_ids_tensor.squeeze(0).tolist())
    logger.info(f"output_text: \n{output_text}")

    # inference 2
    # -----------
    output_token_ids_tensor = generate(
        model = model,
        token_ids = input_token_ids_tensor,
        max_new_tokens = 100,
        eos_token_id = tokenizer.eos_token_id,
        context_length=DotDict(QWEN_CONFIG_06_B).context_length,
    )
    output_text = tokenizer.decode(output_token_ids_tensor.squeeze(0).tolist())
    logger.info(f"output_text: \n{output_text}")

    # inference 3
    # -----------
    start_time = time.time()
    output_token_ids_tensor = generate_qwen3(
        model = model,
        token_ids = input_token_ids_tensor,
        max_new_tokens = 100,
        eos_token_id = tokenizer.eos_token_id,
        context_length=DotDict(QWEN3_CONFIG).context_length,
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
    # -----------
    start_time = time.time()
    output_token_ids_tensor = generate_qwen3(
        model = model,
        token_ids = input_token_ids_tensor,
        max_new_tokens = 100,
        context_length = DotDict(QWEN3_CONFIG).context_length,
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
    # -----------
    start_time = time.time()
    output_token_ids_tensor = generate_qwen3(
        model = model_compiled,
        token_ids = input_token_ids_tensor,
        max_new_tokens = 100,
        context_length = DotDict(QWEN3_CONFIG).context_length,
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
    """
    pass 

if __name__ == "__main__":
    main()
