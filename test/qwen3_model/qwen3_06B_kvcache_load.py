# -*- coding: utf-8 -*-

# ***************************************************
# * File        : qwen3_06B_kvcache_load.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-10-02
# * Version     : 1.0.100217
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
import warnings
warnings.filterwarnings("ignore")

import torch

from test.qwen3_model.load_weights import load_weights_into_qwen

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]
os.environ['LOG_NAME'] = LOGGING_LABEL
from utils.log_util import logger




# 测试代码 main 函数
def main():
    from utils.device import device_setting
    from config.qwen3.model_cfgs import get_cfgs
    from config.qwen3.model_download import download_from_huggingface_from_snapshots
    from models.qwen3.qwen3 import Model
    from layers.tokenizers.qwen3.qwen3_tokenizer import Qwen3Tokenizer
    from layers.inference import generate_qwen3_stream


    # random seed
    torch.manual_seed(0)
    # ------------------------------
    # local path
    # ------------------------------
    llm_model_dir = "./downloaded_models/qwen3_model"

    # ------------------------------
    # device
    # ------------------------------
    device = device_setting(verbose=True)

    # ------------------------------
    # Model and text generation settings
    # ------------------------------
    # model config
    CHOOSE_MODEL = "0.6B"
    logger.info(f"Model config name: {CHOOSE_MODEL}")
    QWEN3_CONFIG = get_cfgs(CHOOSE_MODEL)
    logger.info(f"Model config: {QWEN3_CONFIG}")
    
    # model type
    model_type = "reasoning"
    logger.info(f"Model type: {model_type}")
    # Control base model and the reasoning("thinking") model flag
    if model_type == "base":
        # base model
        USE_REASONING_MODEL = False
        USE_INSTRUCT_MODEL = False
    elif model_type == "reasoning":
        # reasoning("thinking") model
        USE_REASONING_MODEL = True
        USE_INSTRUCT_MODEL = False
    elif model_type == "instruct_without_reasoning":
        # instruct mode(without reasoning)
        USE_REASONING_MODEL = True
        USE_INSTRUCT_MODEL = True
    logger.info(f"Use reasoning model: {USE_REASONING_MODEL}")
    logger.info(f"Use instruct model: {USE_INSTRUCT_MODEL}")

    # ------------------------------
    # text generation settings 
    # ------------------------------
    MAX_NEW_TOKENS = 500
    TEMPERATURE = 0.0
    TOP_K = 1

    # ------------------------------
    # Weights download and loding of the 0.6B model
    # ------------------------------
    # model repo id
    if USE_REASONING_MODEL:
        repo_id = f"Qwen/Qwen3-{CHOOSE_MODEL}"
    else:
        repo_id = f"Qwen/Qwen3-{CHOOSE_MODEL}-Base"

    # model local dir
    local_dir = Path(llm_model_dir).joinpath(Path(repo_id).parts[-1])

    # load model weights
    weights_dict, tokenizer_file_path = download_from_huggingface_from_snapshots(
        repo_id=repo_id, local_dir=local_dir
    )
    
    # model
    # ---------------
    # mode init
    model = Model(cfg=QWEN3_CONFIG)
    # load pretrained model weights
    load_weights_into_qwen(model, QWEN3_CONFIG, weights_dict)
    # move model to devcie
    model.to(device)
    del weights_dict
    # model compile
    # major, minor = map(int, torch.__version__.split(".")[:2])
    # if torch.cuda.is_available():
    #     # if (major, minor) > (2, 8):
    #     #     # This avoids retriggering model recompilations in PyTorch 2.8 and newer
    #     #     # if the model contains code like self.pos = self.pos + 1
    #     #     torch._dynamo.config.allow_unspec_int_on_nn_module = True
    #     model_compiled = torch.compile(model)

    # tokenizer
    # ---------------
    tokenizer = Qwen3Tokenizer(
        tokenizer_file_path=tokenizer_file_path,
        apply_chat_template=USE_REASONING_MODEL,
        add_generation_prompt=USE_REASONING_MODEL,
        add_thinking=not USE_INSTRUCT_MODEL
    )
    logger.info(f"tokenizer.pad_token_id: {tokenizer.pad_token_id}")
    logger.info(f"tokenizer.eos_token_id: {tokenizer.eos_token_id}")
    # ------------------------------
    # model inference
    # ------------------------------
    # prompt
    prompt = "Give me a short introduction to large language models."
    input_token_ids = tokenizer.encode(prompt)
    # text = tokenizer.decode(input_token_ids)
    input_token_ids_tensor = torch.tensor(input_token_ids, device=device).unsqueeze(0)

    for token in generate_qwen3_stream(
        model=model,
        token_idx=input_token_ids_tensor,
        max_new_tokens=MAX_NEW_TOKENS,
        context_length=QWEN3_CONFIG.context_length,
        temperature=TEMPERATURE,
        top_k=TOP_K,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
    ):
        token_id = token.squeeze(0).tolist()
        print(tokenizer.decode(token_id), end="", flush=True)

if __name__ == "__main__":
    main()
