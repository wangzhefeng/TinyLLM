# -*- coding: utf-8 -*-

# ***************************************************
# * File        : kv_cache.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-10-02
# * Version     : 1.0.100203
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

__all__ = ["KVCache", "KVCache_optimized", "trim_kv_cache_window"]

# python libraries
from pathlib import Path

import torch

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


class KVCache:

    def __init__(self, n_layers):
        self.cache = [None] * n_layers
        self.position = 0

    def get(self, layer_idx):
        return self.cache[layer_idx]

    def update(self, layer_idx, value):
        self.cache[layer_idx] = value

    def get_all(self):
        return self.cache

    def advance(self, num_tokens: int) -> int:
        start = self.position
        self.position += num_tokens
        return start

    def reset(self):
        for i in range(len(self.cache)):
            self.cache[i] = None
        self.position = 0


class KVCache_optimized:

    def __init__(self, n_layers, max_len, num_kv_groups, head_dim, device, dtype):
        self.k = [None] * n_layers
        self.v = [None] * n_layers
        self.len = [0] * n_layers
        self.max_len = max_len
        self.num_kv_groups = num_kv_groups
        self.head_dim = head_dim
        self.device = device
        self.dtype = dtype
        self.position = 0

    def advance(self, num_tokens: int) -> int:
        start = self.position
        self.position += num_tokens
        return start

    def allocate(self, layer_idx, b):
        if self.k[layer_idx] is None:
            self.k[layer_idx] = torch.empty(
                b, self.num_kv_groups, self.max_len, self.head_dim,
                device=self.device, dtype=self.dtype
            )
            self.v[layer_idx] = torch.empty(
                b, self.num_kv_groups, self.max_len, self.head_dim,
                device=self.device, dtype=self.dtype
            )
            self.len[layer_idx] = 0

    def append(self, layer_idx, k_new, v_new):
        # 未分配时按首个输入自动分配，避免依赖调用方先调 allocate 的隐式约定
        if self.k[layer_idx] is None:
            self.allocate(layer_idx, k_new.shape[0])
        L = self.len[layer_idx]
        T = k_new.shape[2]
        if L + T > self.max_len:
            raise ValueError(
                f"KV cache overflow at layer {layer_idx}: "
                f"current length {L} + new {T} > max_len {self.max_len}"
            )
        self.k[layer_idx][:, :, L:L+T, :].copy_(k_new)
        self.v[layer_idx][:, :, L:L+T, :].copy_(v_new)
        self.len[layer_idx] = L + T

    def view(self, layer_idx):
        L = self.len[layer_idx]
        return self.k[layer_idx][:, :, :L, :], self.v[layer_idx][:, :, :L, :]

    def reset(self):
        for i in range(len(self.k)):
            self.k[i] = self.v[i] = None
            self.len[i] = 0
        self.position = 0


def trim_kv_cache_window(cache, n_layers, window_size):
    """对 tuple 式 ``KVCache`` 做滑动窗口截断。

    每层只保留最近 ``window_size`` 个 token 的 K/V（沿 seq 维切片），
    将 O(seq_len) 的 cache 增长约束为 O(window_size)。
    与 ``SlidingWindowAttention`` 的内置窗口 cache 不同，本工具用于
    Qwen3 式（tuple cache + 外部 mask）模型在长生成时限制显存。

    Args:
        cache: ``KVCache``，每层存 ``(k, v)``，k/v 形状 (B, H_kv, T, D)
        n_layers: 层数
        window_size: 保留的窗口大小
    """
    for i in range(n_layers):
        kv = cache.get(i)
        if kv is None:
            continue
        k, v = kv
        if k.shape[2] > window_size:
            cache.update(i, (k[:, :, -window_size:, :], v[:, :, -window_size:, :]))




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
