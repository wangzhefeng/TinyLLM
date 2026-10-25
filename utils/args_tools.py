# -*- coding: utf-8 -*-

# ***************************************************
# * File        : args_tools.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-01-13
# * Version     : 1.0.011322
# * Description : argparse helpers used by TinyLLM entry scripts
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

__all__ = [
    "DotDict",
    "print_args_llm",
]

# python libraries
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)

from utils.log_util import logger

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


class DotDict(dict):
    """
    dot.notation access to dictionary attributes
    """
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, dct):
        for key, value in dct.items():
            if hasattr(value, "keys"):
                value = DotDict(value)
            self[key] = value


def _args_to_dict(args):
    """
    argparse.Namespace / DotDict / dict 统一转为 dict
    """
    if isinstance(args, dict):
        return dict(args)

    return vars(args)


def print_args_llm(args):
    """
    打印实验参数。

    与原版逐字段硬编码不同，这里直接遍历参数对象，
    这样 run_gpt2_124M.py / run_llama.py / run_qwen.py 的参数集合
    即使后续增加或重命名字段也不会触发 AttributeError。
    """
    args_dict = _args_to_dict(args)

    logger.info(f"{100 * '-'}")
    logger.info("Args in experiment:")
    logger.info(f"{100 * '-'}")
    for key in sorted(args_dict.keys()):
        value = args_dict[key]
        logger.info(f"  {key + ':':<28}{value}")
    logger.info(f"{100 * '-'}")




# 测试代码 main 函数
def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", type=str, default="gpt2_124M_pretrain")
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()
    print_args_llm(args)

    dct = {"a": 1, "nested": {"b": 2}}
    dot = DotDict(dct)
    logger.info(f"dot.a: {dot.a}, dot.nested.b: {dot.nested.b}")

if __name__ == "__main__":
    main()
