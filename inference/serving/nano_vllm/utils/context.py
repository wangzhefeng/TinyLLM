# -*- coding: utf-8 -*-

# ***************************************************
# * File        : context.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-07-28
# * Version     : 1.0.072816
# * Description : 全局前向上下文(Context)。
# *               engine 在每次 step 前通过 set_context() 写入本 step 的
# *               prefill/decode 元信息，Attention 层通过 get_context() 读取，
# *               从而把 "batch 拼接方式 / KV cache 写入位置" 与模型层解耦。
# * Link        : https://github.com/GeeeekExplorer/nano-vllm
# * Requirement : torch >= 2.0
# ***************************************************

__all__ = [
    "Context",
    "get_context",
    "set_context",
    "reset_context",
]

# python libraries
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)
PKG_ROOT = str(Path(__file__).resolve().parents[1])
if PKG_ROOT in sys.path:
    sys.path.remove(PKG_ROOT)
sys.path.insert(0, PKG_ROOT)
import warnings
warnings.filterwarnings("ignore")

from dataclasses import dataclass

import torch

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


@dataclass
class Context:
    """
    单次前向的调度上下文

    Args:
        is_prefill: 当前 step 是否为 prefill(False 则为 decode)
        cu_seqlens_q / cu_seqlens_k: prefill 时各序列 query/key 的累积长度(varlen 边界)
        max_seqlen_q / max_seqlen_k: prefill 时最长 query/key 长度
        slot_mapping: 每个输入 token 写入 KV cache 的槽位 id(block_id * block_size + offset)
        context_lens: decode 时各序列的上下文长度(含当前 token)
        block_tables: decode 时各序列的 block 表，padding 为 -1，shape [bs, max_num_blocks]
    """
    is_prefill: bool = False
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None


_CONTEXT = Context()


def get_context() -> Context:
    return _CONTEXT


def set_context(is_prefill, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=0, max_seqlen_k=0, slot_mapping=None, context_lens=None, block_tables=None):
    global _CONTEXT
    _CONTEXT = Context(is_prefill, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, context_lens, block_tables)


def reset_context():
    global _CONTEXT
    _CONTEXT = Context()




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
