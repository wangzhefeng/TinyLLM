# -*- coding: utf-8 -*-

# ***************************************************
# * File        : block_manager.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-07-28
# * Version     : 1.0.072816
# * Description : 分页 KV cache 的 block 管理器。
# *               ├── Block: 物理 block，带引用计数与前缀 hash
# *               └── BlockManager: free/used 队列，分配/释放/追加，
# *                   并基于 token hash 支持前缀缓存(prefix caching)复用。
# *               与上游的区别：用内置 hash((prefix, tuple(token_ids)))
# *               替代 xxhash，去掉 numpy 依赖。
# * Link        : https://github.com/GeeeekExplorer/nano-vllm
# * Requirement : 无
# ***************************************************

__all__ = [
    "Block",
    "BlockManager",
]

# python libraries
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

from collections import deque

from .sequence import Sequence

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


class Block:
    """
    KV cache 的一个物理 block(可容纳 block_size 个 token 的 K/V)
    """

    def __init__(self, block_id: int):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:
    """
    分页 block 管理器

    - free_block_ids / used_block_ids: 空闲/占用队列
    - hash_to_block_id: 已满 block 的 token hash -> block_id，用于前缀缓存命中
    """

    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = dict()
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1) -> int:
        """
        计算一个 block 内容的前缀链式 hash：h_i = hash((h_{i-1}, tuple(token_ids)))
        """
        return hash((prefix, tuple(token_ids)))

    def _allocate_block(self) -> int:
        block_id = self.free_block_ids.popleft()
        block = self.blocks[block_id]
        assert block.ref_count == 0
        if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block_id:
            del self.hash_to_block_id[block.hash]
        block.reset()
        self.used_block_ids.add(block_id)
        return block_id

    def _deallocate_block(self, block_id: int):
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def can_allocate(self, seq: Sequence) -> int:
        """
        检查是否有足够空闲 block 容纳 seq；返回前缀缓存命中的 block 数，-1 表示不足
        """
        h = -1
        num_cached_blocks = 0
        num_new_blocks = seq.num_blocks
        for i in range(seq.num_blocks - 1):  # 最后一个 block 通常不满，不参与前缀缓存
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id.get(h, -1)
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                break
            num_cached_blocks += 1
            if block_id in self.used_block_ids:
                num_new_blocks -= 1
        if len(self.free_block_ids) < num_new_blocks:
            return -1
        return num_cached_blocks

    def allocate(self, seq: Sequence, num_cached_blocks: int):
        """
        为 seq 分配 block_table，前缀命中的 block 直接增加引用计数
        """
        assert not seq.block_table
        h = -1
        for i in range(num_cached_blocks):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id[h]
            block = self.blocks[block_id]
            if block_id in self.used_block_ids:
                block.ref_count += 1
            else:
                block.ref_count = 1
                self.free_block_ids.remove(block_id)
                self.used_block_ids.add(block_id)
            seq.block_table.append(block_id)
        for i in range(num_cached_blocks, seq.num_blocks):
            seq.block_table.append(self._allocate_block())
        seq.num_cached_tokens = num_cached_blocks * self.block_size

    def deallocate(self, seq: Sequence):
        """
        释放 seq 占用的全部 block(倒序减少引用计数，归零后归还空闲队列)
        """
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool:
        """
        decode 追加一个 token 是否需要新 block：当已有 token 数恰好为 block 边界 + 1 时需要
        """
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        """
        decode 调度时按需为 seq 追加一个新 block
        """
        if len(seq) % self.block_size == 1:
            seq.block_table.append(self._allocate_block())

    def hash_blocks(self, seq: Sequence):
        """
        把本 step 已填满的 block 写入 hash 表，供后续请求前缀命中
        """
        start = seq.num_cached_tokens // self.block_size
        end = (seq.num_cached_tokens + seq.num_scheduled_tokens) // self.block_size
        if start == end:
            return
        h = self.blocks[seq.block_table[start - 1]].hash if start > 0 else -1
        for i in range(start, end):
            block = self.blocks[seq.block_table[i]]
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block.update(h, token_ids)
            self.hash_to_block_id[h] = block.block_id




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
