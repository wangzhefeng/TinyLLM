"""SWA（滑动窗口注意力）与 KV cache 窗口截断测试（Phase 2 预训练×推断）。"""
import unittest

import torch

from layers.attention import SlidingWindowAttention
from layers.kv_cache import KVCache, trim_kv_cache_window
from layers.masking import create_sliding_window_mask


class TestSlidingWindowMask(unittest.TestCase):
    def test_mask_values(self):
        # window=3：token i 可见 j ∈ [i-2, i]
        m = create_sliding_window_mask(5, 3)
        self.assertFalse(m[0, 0])               # 自身
        self.assertTrue(m[0, 1])                # 未来
        self.assertFalse(m[2, 0])               # 距离 2 < 3，可见
        self.assertFalse(m[2, 1])
        self.assertTrue(m[3, 0])                # 距离 3 >= 3，屏蔽
        self.assertFalse(m[4, 2])               # 距离 2，可见
        self.assertTrue(m[4, 1])                # 距离 3，屏蔽


class TestSlidingWindowAttention(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.attn = SlidingWindowAttention(
            d_model=16, d_out=16, n_heads=2, context_length=64,
            window_size=4, dropout=0.0,
        ).eval()

    def test_out_of_window_tokens_do_not_matter(self):
        """SWA 核心性质：窗口之外的历史 token 不影响输出。"""
        x1 = torch.randn(1, 8, 16)
        x2 = x1.clone()
        x2[:, :1, :] = torch.randn(1, 1, 16)     # 只篡改位置 0
        with torch.no_grad():
            o1 = self.attn(x1)
            o2 = self.attn(x2)
        # window=4：token i 只看 [i-3, i]，位置 0 仅被 token 0..3 看到
        # 因此最后 4 个 token（4..7）的输出必须不受影响
        torch.testing.assert_close(o1[:, 4:], o2[:, 4:])

    def test_cache_matches_no_cache(self):
        """逐 token 增量（cache 窗口左移）与整段前向一致。"""
        x = torch.randn(1, 10, 16)
        with torch.no_grad():
            full = self.attn(x)
            self.attn.reset_cache()
            outs = []
            for t in range(10):
                out_t = self.attn(x[:, t:t+1, :], use_cache=True)
                outs.append(out_t)
        inc = torch.cat(outs, dim=1)
        torch.testing.assert_close(full, inc, atol=1e-5, rtol=1e-4)

    def test_cache_size_bounded(self):
        """cache 恒不超过 window_size（显存 O(window)）。"""
        x = torch.randn(1, 12, 16)
        with torch.no_grad():
            for t in range(12):
                self.attn(x[:, t:t+1, :], use_cache=True)
        self.assertEqual(self.attn.cache_k.shape[2], self.attn.window_size)


class TestTrimKVCacheWindow(unittest.TestCase):
    def test_trims_to_window(self):
        cache = KVCache(n_layers=2)
        k = torch.randn(1, 2, 10, 8)
        v = torch.randn(1, 2, 10, 8)
        cache.update(0, (k, v))
        cache.update(1, (k.clone(), v.clone()))
        trim_kv_cache_window(cache, n_layers=2, window_size=4)
        for i in range(2):
            tk, tv = cache.get(i)
            self.assertEqual(tk.shape[2], 4)
            torch.testing.assert_close(tk, k[:, :, -4:, :])
        # 短于窗口的不截断
        cache2 = KVCache(n_layers=1)
        cache2.update(0, (k[:, :, :3], v[:, :, :3]))
        trim_kv_cache_window(cache2, n_layers=1, window_size=4)
        self.assertEqual(cache2.get(0)[0].shape[2], 3)


if __name__ == "__main__":
    unittest.main()
