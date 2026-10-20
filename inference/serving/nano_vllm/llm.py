# -*- coding: utf-8 -*-

# ***************************************************
# * File        : llm.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2026-10-20
# * Version     : 1.0.102020
# * Description : LLM: 对齐上游 nanovllm/llm.py 的便捷入口。
# *               继承 LLMEngine，允许直接以关键字参数构造 Config。
# * Link        : https://github.com/GeeeekExplorer/nano-vllm
# * Requirement : torch >= 2.0
# ***************************************************

__all__ = [
    "LLM",
]

# python libraries
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

from .config import Config
from .engine.llm_engine import LLMEngine

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


class LLM(LLMEngine):
    """
    便捷入口(对齐上游 nanovllm.llm.LLM)：
    可直接传入 Config，或以关键字参数同时给出 Config 字段与引擎参数

    Example:
        llm = LLM(vocab_size=64, hidden_size=128, eos=0)
        llm = LLM(config, tokenizer=tokenizer, verbose=True)
    """

    def __init__(self, config: Config | None = None, **kwargs):
        engine_keys = ("model", "tokenizer", "verbose")
        engine_kwargs = {k: kwargs.pop(k) for k in engine_keys if k in kwargs}
        if config is None:
            config = Config(**kwargs)
        else:
            assert not kwargs, f"config 与 Config 字段参数不可混用: {list(kwargs)}"
        super().__init__(config, **engine_kwargs)
