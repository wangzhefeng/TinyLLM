# -*- coding: utf-8 -*-

# ***************************************************
# * File        : weight_loading.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2026-10-18
# * Version     : 1.0.101800
# * Description : 内存高效的权重加载（LLMs-from-scratch ch05/08）：
# *               mmap 加载避免双倍内存；可选 meta-device + assign 零拷贝注入
# * Link        : https://github.com/rasbt/LLMs-from-scratch/tree/main/ch05/08_bonus_memory_efficient_weight_loading
# * Requirement : torch >= 2.1
# ***************************************************

__all__ = [
    "load_state_dict_mmap",
    "load_weights_into_model",
]

# python libraries
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)
from typing import Union

import torch
import torch.nn as nn

from utils.log_util import logger

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


def load_state_dict_mmap(path: Union[str, Path]) -> dict:
    """
    以内存映射方式加载 checkpoint，避免整个文件先读入内存。

    自动解包常见封装：
    - {"model_state_dict": ...} / {"model": ...}（tools/checkpoint、EarlyStopping 格式）
    - 纯 state_dict
    """
    path = Path(path).expanduser()
    try:
        payload = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    except TypeError:
        # 旧版本 torch 不支持 mmap 参数
        payload = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(payload, dict):
        for key in ("model_state_dict", "model"):
            if key in payload and isinstance(payload[key], dict):
                return payload[key]
    if not isinstance(payload, dict):
        raise TypeError(f"unsupported checkpoint format at {path}: {type(payload)}")
    return payload


def load_weights_into_model(model: nn.Module,
                            source: Union[str, Path, dict],
                            strict: bool = True,
                            strip_module_prefix: bool = True) -> nn.Module:
    """
    把权重注入已有模型实例。

    - source 为路径时走 mmap 加载（内存高效）；
    - strip_module_prefix=True 时自动剥掉 DP/DDP 的 "module." 前缀；
    - 大模型可先用 `with torch.device("meta"):` 构建模型再以
      ``model.load_state_dict(sd, assign=True)`` 零拷贝注入（本函数不对 meta
      模型兜底，调用方需自行选择 assign=True 的场景）。

    Returns:
        加载后的模型（就地修改）
    """
    if isinstance(source, (str, Path)):
        state_dict = load_state_dict_mmap(source)
    elif isinstance(source, dict):
        state_dict = source
    else:
        raise TypeError(f"unsupported source type: {type(source)}")
    # 剥掉 "module." 前缀
    if strip_module_prefix and any(k.startswith("module.") for k in state_dict):
        state_dict = {
            k[len("module."):] if k.startswith("module.") else k: v
            for k, v in state_dict.items()
        }
    model.load_state_dict(state_dict, strict=strict)
    logger.info(f"Weights loaded into {type(model).__name__} (strict={strict})")

    return model




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
