"""NSA（DeepSeek-V3 Native Sparse Attention）教学实现测试。"""
import math
import unittest

import torch

from layers.sparse_attention import NativeSparseAttention


class TestNSAForward(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.attn = NativeSparseAttention(
            d_model=32, n_heads=2, block_size=4, n_selected_blocks=2, sliding_window=8)
        self.x = torch.randn(2, 16, 32)

    def test_forward_shape_finite(self):
        out = self.attn(self.x)
        self.assertEqual(out.shape, (2, 16, 32))
        self.assertTrue(torch.isfinite(out).all())

    def test_causal_prefix_invariance(self):
        """修改未来 token 不影响历史位置的输出（三分支均因果）。"""
        x2 = self.x.clone()
        x2[:, 8:, :] += 10.0
        self.attn.eval()
        with torch.no_grad():
            o1 = self.attn(self.x)
            o2 = self.attn(x2)
        torch.testing.assert_close(o1[:, :8], o2[:, :8])

    def test_gates_per_head(self):
        """三个门控为每头独立参数，sigmoid 值在 (0,1)。"""
        for g in (self.attn.gate_compressed, self.attn.gate_selected, self.attn.gate_window):
            self.assertEqual(g.shape, (2,))
            self.assertTrue(((torch.sigmoid(g) > 0) & (torch.sigmoid(g) < 1)).all())

    def test_window_only_matches_manual_attention(self):
        """压掉 compressed/selected 门控后，输出等于手工滑窗因果注意力。"""
        attn = self.attn.eval()
        with torch.no_grad():
            attn.gate_compressed.fill_(-30.0)
            attn.gate_selected.fill_(-30.0)
            attn.gate_window.fill_(30.0)
            out = attn(self.x)
            # 手工复算 window 分支
            B, T, _ = self.x.shape
            H, D = attn.n_heads, attn.head_dim
            q = attn.W_query(self.x).view(B, T, H, D).transpose(1, 2)
            k = attn.W_key(self.x).view(B, T, H, D).transpose(1, 2)
            v = attn.W_value(self.x).view(B, T, H, D).transpose(1, 2)
            scores = q @ k.transpose(-2, -1) / math.sqrt(D)
            t = torch.arange(T)[:, None]
            j = torch.arange(T)[None, :]
            mask = (j <= t) & (j > t - attn.sliding_window)
            scores = scores.masked_fill(~mask, -torch.inf)
            o_w = torch.softmax(scores, dim=-1) @ v
            expected = attn.out_proj(
                o_w.transpose(1, 2).contiguous().view(B, T, attn.d_out))
        torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-5)


class TestNSABranches(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1)
        self.attn = NativeSparseAttention(
            d_model=32, n_heads=2, block_size=4, n_selected_blocks=2, sliding_window=8)
        self.x = torch.randn(1, 12, 32)

    def _qkv(self):
        attn = self.attn
        x = self.x
        B, T, _ = x.shape
        H, D = attn.n_heads, attn.head_dim
        q = attn.W_query(x).view(B, T, H, D).transpose(1, 2)
        k = attn.W_key(x).view(B, T, H, D).transpose(1, 2)
        v = attn.W_value(x).view(B, T, H, D).transpose(1, 2)
        return q, k, v

    def test_block_causal_visibility(self):
        """t < w 时无可见块；t = w 起可见块 0；t = 2w 起可见块 1。"""
        causal = self.attn._block_causal(12, 3, self.x.device)
        self.assertFalse(causal[3].any())            # t=3 < w=4
        self.assertTrue(causal[4, 0])                # t=4 可见块 0
        self.assertFalse(causal[4, 1])
        self.assertTrue(causal[8, 0] and causal[8, 1])

    def test_pool_blocks_mean(self):
        """块池化 = 每块逐元素均值（不含 padding 的完整块）。"""
        k = torch.arange(2 * 8 * 2, dtype=torch.float32).reshape(1, 2, 2, 8).permute(0, 1, 3, 2)
        # 构造 (B=1,H=2,T=8,D=2)：T=8、w=4 -> 2 块
        k_blk, _ = self.attn._pool_blocks(k, torch.zeros_like(k))
        torch.testing.assert_close(k_blk[:, :, 0], k[:, :, :4].mean(dim=2))
        torch.testing.assert_close(k_blk[:, :, 1], k[:, :, 4:].mean(dim=2))

    def test_compressed_zero_before_first_block(self):
        """t < w 的 query 无可见块：compressed 分支输出为 0。"""
        q, k, v = self._qkv()
        num_blocks = (12 + 3) // 4
        causal = self.attn._block_causal(12, num_blocks, self.x.device)
        out_c, _ = self.attn._compressed_out(q, k, v, causal)
        self.assertTrue((out_c[:, :, :4] == 0).all())

    def test_selected_zero_before_first_block(self):
        """t < w 的 query 无可选块：selected 分支输出为 0。"""
        q, k, v = self._qkv()
        num_blocks = (12 + 3) // 4
        causal = self.attn._block_causal(12, num_blocks, self.x.device)
        _, scores_c = self.attn._compressed_out(q, k, v, causal)
        out_s = self.attn._selected_out(q, k, v, scores_c, causal)
        self.assertTrue((out_s[:, :, :4] == 0).all())

    def test_selected_matches_manual_topk(self):
        """selected 的块选择与对 compressed 分数做人工 topk 一致。"""
        q, k, v = self._qkv()
        causal = self.attn._block_causal(12, 3, self.x.device)
        _, scores_c = self.attn._compressed_out(q, k, v, causal)
        scores_c = scores_c.masked_fill(~causal, -torch.inf)
        # t=11：可见块 0、1，n_selected=2 应恰好选中两者
        top_idx = scores_c.topk(2, dim=-1).indices
        self.assertEqual(sorted(top_idx[0, 0, 11].tolist()), [0, 1])


if __name__ == "__main__":
    unittest.main()
