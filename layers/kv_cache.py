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

__all__ = []

# python libraries
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)
import warnings
warnings.filterwarnings("ignore")

import torch

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


class KVCache:

    def __init__(self, n_layers):
        self.cache = [None] * n_layers

    def get(self, layer_idx):
        return self.cache[layer_idx]

    def update(self, layer_idx, value):
        self.cache[layer_idx] = value

    def get_all(self):
        return self.cache

    def reset(self):
        for i in range(len(self.cache)):
            self.cache[i] = None


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
        L = self.len[layer_idx]
        T = k_new.shape[2]
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




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
