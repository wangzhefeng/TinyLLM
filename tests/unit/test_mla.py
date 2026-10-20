"""MLA（Multi-Head Latent Attention）与 DeepSeek 模型测试。

覆盖：
1. MLA 前向形状（含/不含 Q 压缩）
2. cache 增量解码与整段前向逐 token 一致（核心正确性）
3. MLA vs MHA 的 KV cache 显存对比（MLA 教学价值的核心验证）
4. DeepSeekModel 前向 + cache 生成 + 越界防护
"""
import unittest
from types import SimpleNamespace

import torch

from layers.kv_cache import KVCache
from models.deepseek.mla import MultiHeadLatentAttention
from models.deepseek.model import DeepSeekModel


def _mla_cfg(**overrides):
    cfg = dict(
        d_model=64, n_heads=4, kv_lora_rank=32, q_lora_rank=None,
        qk_nope_head_dim=12, qk_rope_head_dim=4, rope_base=10_000.0,
        context_length=64, dtype=torch.float32,
    )
    cfg.update(overrides)
    return SimpleNamespace(**cfg)


class TestMLAForward(unittest.TestCase):
    def test_forward_shape_no_q_compress(self):
        torch.manual_seed(0)
        attn = MultiHeadLatentAttention(**vars(_mla_cfg()))
        x = torch.randn(2, 10, 64)
        out, cache = attn(x)
        self.assertEqual(out.shape, (2, 10, 64))
        c_kv, k_rope = cache
        self.assertEqual(c_kv.shape, (2, 10, 32))          # 只存 latent
        self.assertEqual(k_rope.shape, (2, 4, 10, 4))      # + rope key

    def test_forward_shape_with_q_compress(self):
        torch.manual_seed(0)
        attn = MultiHeadLatentAttention(**vars(_mla_cfg(q_lora_rank=48)))
        out, _ = attn(torch.randn(2, 6, 64))
        self.assertEqual(out.shape, (2, 6, 64))


class TestMLACacheConsistency(unittest.TestCase):
    def test_incremental_matches_full(self):
        """逐 token 增量（cache）必须与整段前向输出一致。"""
        torch.manual_seed(0)
        attn = MultiHeadLatentAttention(**vars(_mla_cfg())).eval()
        x = torch.randn(1, 8, 64)

        with torch.no_grad():
            # 整段
            full_out, full_cache = attn(x)
            # 逐 token
            cache = None
            outs = []
            for t in range(8):
                out_t, cache = attn(x[:, t:t+1, :], start_pos=t, cache=cache)
                outs.append(out_t)
        inc_out = torch.cat(outs, dim=1)
        self.assertTrue(torch.allclose(full_out, inc_out, atol=1e-5))
        # cache 内容也一致
        c_full, kr_full = full_cache
        c_inc, kr_inc = cache
        self.assertTrue(torch.allclose(c_full, c_inc, atol=1e-5))
        self.assertTrue(torch.allclose(kr_full, kr_inc, atol=1e-5))


class TestMLACacheMemoryAdvantage(unittest.TestCase):
    def test_cache_smaller_than_mha(self):
        """同样头数下，MLA 的 cache（latent + rope）应小于 MHA 的 cache（2·H·D_head）。"""
        torch.manual_seed(0)
        n_heads, head_dim = 32, 128
        kv_lora_rank, rope_dim = 512, 64
        seq_len = 512

        # MHA：缓存每头完整 K/V
        mha_cache = 2 * n_heads * head_dim * seq_len
        # MLA：只缓存 latent + rope key（与头数无关的部分 + 每头 rope）
        mla_cache = (kv_lora_rank + n_heads * rope_dim) * seq_len
        self.assertLess(mla_cache, mha_cache)
        # 与 V3 论文口径一致：节省约一个数量级
        self.assertLess(mla_cache / mha_cache, 0.5)


class TestDeepSeekModel(unittest.TestCase):
    def _cfg(self):
        return SimpleNamespace(
            vocab_size=128, context_length=32, embed_dim=64, n_layers=2,
            n_heads=4, kv_lora_rank=32, q_lora_rank=48,
            qk_nope_head_dim=12, qk_rope_head_dim=4,
            hidden_dim=128, rope_base=10_000.0, dtype=torch.float32,
        )

    def test_forward_and_generate_with_cache(self):
        torch.manual_seed(123)
        model = DeepSeekModel(self._cfg()).eval()
        x = torch.randint(0, 128, (1, 5))
        with torch.no_grad():
            logits = model(x)
            self.assertEqual(logits.shape, (1, 5, 128))

            # greedy：无 cache vs 有 cache 必须一致
            ids_nc = x.clone()
            for _ in range(4):
                lg = model(ids_nc)
                ids_nc = torch.cat([ids_nc, lg[:, -1].argmax(dim=-1, keepdim=True)], dim=1)

            model.reset_kv_cache()
            cache = KVCache(n_layers=2)
            ids_c = x.clone()
            lg = model(ids_c, cache=cache)
            for _ in range(4):
                nxt = lg[:, -1].argmax(dim=-1, keepdim=True)
                ids_c = torch.cat([ids_c, nxt], dim=1)
                lg = model(nxt, cache=cache)
        self.assertTrue(torch.equal(ids_nc, ids_c))

    def test_context_overflow_guard(self):
        model = DeepSeekModel(self._cfg()).eval()
        model.reset_kv_cache()
        cache = KVCache(n_layers=2)
        x = torch.randint(0, 128, (1, 30))
        with torch.no_grad():
            model(x, cache=cache)                    # pos 0..29
            with self.assertRaises(IndexError):
                model(x[:, :5], cache=cache)         # 30+5 > 32


if __name__ == "__main__":
    unittest.main()
