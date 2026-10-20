# -*- coding: utf-8 -*-

# ***************************************************
# * File        : sequence.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-07-28
# * Version     : 1.0.072816
# * Description : Sequence: 一条生成请求的运行时状态。
# *               维护 token_ids / block_table / 调度计数(num_cached_tokens,
# *               num_scheduled_tokens) / 状态机(WAITING -> RUNNING -> FINISHED)。
# * Link        : https://github.com/GeeeekExplorer/nano-vllm
# * Requirement : torch >= 2.0
# ***************************************************

__all__ = [
    "Sequence",
    "SequenceStatus",
]

# python libraries
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

from copy import copy
from enum import Enum, auto
from itertools import count

from ..sampling_params import SamplingParams

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


class Sequence:
    """
    一条生成请求

    - block_size 为类属性，由 LLMEngine 在启动时从 Config 写入
    - num_cached_tokens: 已经过模型前向处理(KV 已写入 cache)的 token 数
    - num_scheduled_tokens: 当前 step 被调度处理的 token 数
    """
    block_size = 256
    counter = count()

    def __init__(self, token_ids: list[int], sampling_params: SamplingParams = None):
        sampling_params = sampling_params or SamplingParams()
        self.seq_id = next(Sequence.counter)
        self.status = SequenceStatus.WAITING
        self.token_ids = copy(token_ids)
        self.last_token = token_ids[-1]
        self.num_tokens = len(self.token_ids)
        self.num_prompt_tokens = len(token_ids)
        self.num_cached_tokens = 0
        self.num_scheduled_tokens = 0
        self.is_prefill = True
        self.block_table = []
        # 采样参数透传
        self.temperature = sampling_params.temperature
        self.top_k = sampling_params.top_k
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos

    def __len__(self):
        return self.num_tokens

    def __getitem__(self, key):
        return self.token_ids[key]

    @property
    def is_finished(self):
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self):
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self):
        return self.token_ids[:self.num_prompt_tokens]

    @property
    def completion_token_ids(self):
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_blocks(self):
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self):
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def block(self, i):
        assert 0 <= i < self.num_blocks
        return self.token_ids[i * self.block_size: (i + 1) * self.block_size]

    def append_token(self, token_id: int):
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1

    def __getstate__(self):
        """
        pickle 序列化(多进程 TP 时经 shm 传给 worker)：
        prefill 阶段传完整 token_ids，decode 阶段只传 last_token，
        避免每个 step 重复传输整条序列。采样参数/状态由 rank0 主进程持有，
        worker 只需要前向所需字段。
        """
        last_state = self.last_token if not self.is_prefill else self.token_ids
        return (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.num_scheduled_tokens, self.block_table, last_state)

    def __setstate__(self, state):
        self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.num_scheduled_tokens, self.block_table, last_state = state
        if isinstance(last_state, list):
            self.token_ids = last_state
            self.last_token = self.token_ids[-1]
        else:
            self.token_ids = []
            self.last_token = last_state




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
