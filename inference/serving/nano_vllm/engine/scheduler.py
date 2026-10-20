# -*- coding: utf-8 -*-

# ***************************************************
# * File        : scheduler.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-07-28
# * Version     : 1.0.072816
# * Description : 调度器：实现 continuous batching。
# *               - schedule(): 优先从 waiting 队列做 prefill(支持分块 prefill)，
# *                 否则从 running 队列做 decode；显存不足时抢占(preempt)队尾序列
# *               - postprocess(): 写回采样 token，判断结束(eos / max_tokens)
# *               与上游 nanovllm/engine/scheduler.py 逻辑一致。
# * Link        : https://github.com/GeeeekExplorer/nano-vllm
# * Requirement : 无
# ***************************************************

__all__ = [
    "Scheduler",
]

# python libraries
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

from collections import deque

from ..config import Config
from .block_manager import BlockManager
from .sequence import Sequence, SequenceStatus

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


class Scheduler:
    """
    连续批处理调度器

    每个 step 的决策：
    1. waiting 非空 -> prefill(按 max_num_batched_tokens 分块，按 max_num_seqs 限制条数)
    2. waiting 为空 -> decode(running 中所有序列各前进 1 个 token)
    """

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_size = config.kvcache_block_size
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        scheduled_seqs = []
        num_batched_tokens = 0

        # prefill
        while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.waiting[0]
            remaining = self.max_num_batched_tokens - num_batched_tokens
            if remaining == 0:
                break
            if not seq.block_table:
                num_cached_blocks = self.block_manager.can_allocate(seq)
                if num_cached_blocks == -1:
                    break
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
            else:
                num_tokens = seq.num_tokens - seq.num_cached_tokens
            if remaining < num_tokens and scheduled_seqs:
                # 只允许多序列批中的第一条做 chunked prefill，避免队头阻塞
                break
            if not seq.block_table:
                self.block_manager.allocate(seq, num_cached_blocks)
            seq.num_scheduled_tokens = min(num_tokens, remaining)
            num_batched_tokens += seq.num_scheduled_tokens
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
            scheduled_seqs.append(seq)

        if scheduled_seqs:
            return scheduled_seqs, True

        # decode
        while self.running and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                seq.num_scheduled_tokens = 1
                seq.is_prefill = False
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        assert scheduled_seqs
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        """
        抢占：释放 seq 的 KV block，重新放回 waiting 队首(下次从头 prefill)
        """
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool):
        """
        前向完成后：更新已缓存计数、追加密采样的 token、判断结束条件
        """
        for seq, token_id in zip(seqs, token_ids):
            self.block_manager.hash_blocks(seq)
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            if is_prefill and seq.num_cached_tokens < seq.num_tokens:
                continue  # chunked prefill 尚未完成整条 prompt
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
