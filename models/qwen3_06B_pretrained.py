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

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]
os.environ['LOG_NAME'] = LOGGING_LABEL
from utils.log_util import logger


# device
device = device_setting(verbose=True)

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
logger.info(f"model: \n{model}")




# 测试代码 main 函数
def main():
    
    pass

if __name__ == "__main__":
    main()
