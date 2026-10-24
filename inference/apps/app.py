# -*- coding: utf-8 -*-

# ***************************************************
# * File        : app.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2026-10-20
# * Version     : 1.0.102000
# * Description : Chainlit 聊天 UI（合并原 app_orig.py / app_own.py）。
#               用环境变量 GPT2_WEIGHTS=orig|own（默认 own）选择权重来源：
#                 - orig：官方 GPT-2 124M 权重（下载到 downloaded_models/gpt2）
#                 - own ：自训 checkpoint（环境变量 GPT2_CHECKPOINT，默认仓根下 model.pth）
# * Link        : link
# * Requirement : pip install chainlit（可选依赖，不在主 requirements 中）；
#               orig 模式另需 tensorflow（仅读官方 TF checkpoint 时需要）
# ***************************************************

__all__ = []

# python libraries
import os
import sys
from pathlib import Path
# 仓根路径由本文件位置推导（inference/apps/app.py -> 上两级），
# 不依赖启动时的 cwd，保证从任意目录运行都能正确 import 与定位权重
ROOT = str(Path(__file__).resolve().parents[2])
if ROOT not in sys.path:
    sys.path.append(ROOT)
import warnings
warnings.filterwarnings("ignore")

import torch

# chainlit 为可选依赖：未安装时模块仍可导入（便于静态检查/复用），仅 UI handler 不注册
try:
    import chainlit
except ImportError:
    chainlit = None
    print("[app.py] 未安装 chainlit，无法启动聊天 UI。请先 pip install chainlit")

from layers.tokenizers.tokenization import (
    choose_tokenizer,
    text_to_token_ids,
    token_ids_to_text,
)
from models.gpt2.gpt2_124M import Model
from inference.generate_core import generate
from utils.args_tools import DotDict
from utils.device import device_setting

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]
os.environ['LOG_NAME'] = LOGGING_LABEL
from utils.log_util import logger

# 权重来源：orig = 官方 GPT-2 124M 权重；own = 自训 checkpoint
WEIGHTS_SOURCE = os.environ.get("GPT2_WEIGHTS", "own").lower()
if WEIGHTS_SOURCE not in ("orig", "own"):
    raise ValueError(f"GPT2_WEIGHTS must be 'orig' or 'own', got: {WEIGHTS_SOURCE}")

# device
device = device_setting(verbose=True)


def get_model_and_tokenizer():
    # model config (gpt2-small 124M)
    model_config = {
        "vocab_size": 50257,     # Vocabular size
        "context_length": 256,  # Context length
        "max_new_tokens": 50,   # Maximum new tokens to generate
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
        "use_cache": True,  # GPT-2 已实现 KV cache（models/gpt2/gpt2_124M.py），开启可加速逐 token 生成
        "tokenizer_model": "tiktoken_gpt2_bpe",
    }
    model_config = DotDict(model_config)

    # tokenizer
    tokenizer = choose_tokenizer(tokenizer_model = "tiktoken_gpt2_bpe")

    # model
    if WEIGHTS_SOURCE == "orig":
        # 官方 GPT-2 124M 权重（懒加载：tensorflow 仅在此模式下需要）
        from models.io.gpt2.openai_gpt2_weights_load import (
            download_and_load_gpt2,
            load_weights_download,
        )
        settings, params = download_and_load_gpt2(
            model_size="124M", models_dir=str(Path(ROOT) / "downloaded_models" / "gpt2")
        )
        model = Model(model_config)
        load_weights_download(model, params)
    else:
        # 自训 checkpoint（路径可用环境变量 GPT2_CHECKPOINT 覆盖，默认仓根下 model.pth）
        model_path = Path(os.environ.get("GPT2_CHECKPOINT", str(Path(ROOT) / "model.pth")))
        if not model_path.exists():
            logger.info(f"Could not find the {model_path} file.")
            sys.exit()
        checkpoint = torch.load(model_path, weights_only=True)
        model = Model(model_config)
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    return tokenizer, model, model_config


if chainlit is not None:
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
            token_ids=text_to_token_ids(
                message.content, 
                tokenizer_model=model_config.tokenizer_model
            ).to(device),
            max_new_tokens=model_config.max_new_tokens,
            context_length=model_config.context_length,
            temperature=model_config.temperature,
            top_k=model_config.top_k,
            eos_token_id=model_config.eos_id,
            use_cache=model_config.use_cache,
        )
        text = token_ids_to_text(token_ids, tokenizer_model=model_config.tokenizer_model)

        # Returns the model response to the interface
        await chainlit.Message(content=f"{text}").send()
else:
    def main(*args, **kwargs):
        raise RuntimeError(
            "chainlit 未安装，无法启动聊天 UI。请先 pip install chainlit，"
            "再用 inference/scripts/run_app.sh [orig|own] 启动。"
        )

if __name__ == "__main__":
    main()
