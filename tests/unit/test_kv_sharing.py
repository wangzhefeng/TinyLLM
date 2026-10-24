"""跨层 KV 共享（CLA）测试：SharedKVCache 与 Qwen3 模型接线。"""
import unittest
from types import SimpleNamespace

import torch

from layers.kv_cache import KVCache, SharedKVCache
from models.qwen3.qwen3 import Model


def _cfg(**overrides):
    cfg = dict(
        vocab_size=1000,
        context_length=64,
        embed_dim=64,
        n_heads=4,
        n_kv_groups=2,
        n_layers=4,
        head_dim=None,
        qk_norm=True,
        num_experts=0,
        hidden_dim=128,
        rope_base=10_000.0,
        dtype=torch.float32,
        kv_sharing_span=1,
    )
    cfg.update(overrides)
    return SimpleNamespace(**cfg)


class TestSharedKVCache(unittest.TestCase):
    def test_group_mapping(self):
        """n_layers=4、span=2：层 0/1 -> 组 0，层 2/3 -> 组 1。"""
        cache = SharedKVCache(n_layers=4, share_span=2)
        self.assertEqual(cache.n_groups, 2)
        k0 = torch.zeros(1, 2, 3, 8)
        cache.update(0, k0)
        # 同组层读到同一槽位
        self.assertIs(cache.get(0), k0)
        self.assertIs(cache.get(1), k0)
        k2 = torch.ones(1, 2, 3, 8)
        cache.update(3, k2)
        self.assertIs(cache.get(2), k2)
        # reset 清空
        cache.reset()
        self.assertIsNone(cache.get(0))
        self.assertEqual(cache.position, 0)

    def test_span1_degenerates_to_kv_cache(self):
        cache = SharedKVCache(n_layers=4, share_span=1)
        self.assertEqual(cache.n_groups, 4)
        cache.update(2, "x")
        self.assertEqual(cache.get(2), "x")
        self.assertIsNone(cache.get(1))

    def test_invalid_span(self):
        with self.assertRaises(ValueError):
            SharedKVCache(n_layers=4, share_span=0)


class TestQwen3KVSharing(unittest.TestCase):
    def test_shared_projections_identity(self):
        torch.manual_seed(0)
        model = Model(_cfg(kv_sharing_span=2))
        # 组内第 2 层与第 1 层共享 KV 投影与 k_norm
        self.assertIs(model.trf_blocks[1].attn.W_key, model.trf_blocks[0].attn.W_key)
        self.assertIs(model.trf_blocks[1].attn.W_value, model.trf_blocks[0].attn.W_value)
        self.assertIs(model.trf_blocks[1].attn.k_norm, model.trf_blocks[0].attn.k_norm)
        # 组内上层只复用不追加；底层正常追加
        self.assertTrue(model.trf_blocks[1].attn.reuse_kv_only)
        self.assertFalse(model.trf_blocks[0].attn.reuse_kv_only)
        # 组间不共享
        self.assertIsNot(model.trf_blocks[2].attn.W_key, model.trf_blocks[0].attn.W_key)

    def test_param_count_reduced(self):
        torch.manual_seed(0)
        shared = Model(_cfg(kv_sharing_span=2))
        plain = Model(_cfg(kv_sharing_span=1))
        shared_params = sum(p.numel() for p in shared.parameters())
        plain_params = sum(p.numel() for p in plain.parameters())
        # 每组省 1 份 W_key/W_value/k_norm（2 组）
        self.assertLess(shared_params, plain_params)

    def test_new_cache_type(self):
        torch.manual_seed(0)
        model = Model(_cfg(kv_sharing_span=2))
        cache = model.new_cache()
        self.assertIsInstance(cache, SharedKVCache)
        self.assertEqual(cache.n_groups, 2)
        plain = Model(_cfg(kv_sharing_span=1))
        self.assertIsInstance(plain.new_cache(), KVCache)

    def test_cached_two_stage_forward(self):
        """prefill + 增量两段式带 cache 前向正常，组内 K/V 长度正确。"""
        torch.manual_seed(0)
        model = Model(_cfg(kv_sharing_span=2)).eval()
        cache = model.new_cache()
        x1 = torch.randint(0, 1000, (1, 6))
        x2 = torch.randint(0, 1000, (1, 3))
        with torch.no_grad():
            out1 = model(x1, cache=cache)
            out2 = model(x2, cache=cache)
        self.assertEqual(out1.shape, (1, 6, 1000))
        self.assertEqual(out2.shape, (1, 3, 1000))
        # 每组只有底层写入：组 0 的 K/V 长度 = 已处理 token 数
        k, v = cache.get(0)
        self.assertEqual(k.shape[2], 9)

    def test_span1_forward_unchanged(self):
        """缺省 span=1：各层投影独立，无 reuse 标记，行为不变。"""
        torch.manual_seed(0)
        model = Model(_cfg()).eval()
        for blk in model.trf_blocks:
            self.assertFalse(getattr(blk.attn, "reuse_kv_only", False))
        x = torch.randint(0, 1000, (2, 8))
        with torch.no_grad():
            out = model(x)
        self.assertEqual(out.shape, (2, 8, 1000))


if __name__ == "__main__":
    unittest.main()
