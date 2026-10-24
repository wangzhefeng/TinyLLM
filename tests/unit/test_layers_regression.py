"""Phase 0 回归测试：覆盖 docs/60-reference/layers-models-audit/bugs.md 中已修复的 P0/P1。

每项测试对应一条已修复的 bug，防止重复实现再次 drift。
"""
import unittest
from types import SimpleNamespace

import torch

from layers.activation import SwiGLU
from layers.attention import (
    MultiHeadAttention,
    MultiHeadAttentionCombinedQKV,
    GroupedQueryAttention_Qwen3_optimized,
)
from layers.embedding.text_embedding import AbsolutePositionEmbeddings
from layers.feed_forward import FeedForwardGELU, FeedForwardSiLU, FeedForwardSwiGLU
from layers.kv_cache import KVCache_optimized
from layers.normalization.rms_norm import RMSNorm
from layers.position_encoding.FixPE import PositionalEncoding, PositionalEncodingWithDropout
from models.gpt2.gpt2_124M import Model as GPT2Model


def _gpt2_cfg():
    return SimpleNamespace(
        vocab_size=128, context_length=32, embed_dim=32, d_ff=64,
        n_heads=4, n_layers=2, dropout=0.0, qkv_bias=True,
        dtype=torch.float32, is_train=False,
    )


class TestAttentionScale(unittest.TestCase):
    """bugs.md #1: CombinedQKV 的 softmax 缩放必须是 1/sqrt(head_dim)。"""

    def test_combined_qkv_matches_mha(self):
        torch.manual_seed(0)
        cfg = dict(d_model=16, d_out=16, n_heads=4, context_length=8, dropout=0.0, qkv_bias=False)
        mha = MultiHeadAttention(**cfg).eval()
        combined = MultiHeadAttentionCombinedQKV(**cfg).eval()
        # 把 MHA 的 Q/K/V 权重拷进 CombinedQKV 的融合 qkv
        with torch.no_grad():
            qkv_w = torch.cat([
                mha.W_query.weight, mha.W_key.weight, mha.W_value.weight,
            ], dim=0)
            combined.qkv.weight.copy_(qkv_w)
            combined.out_proj.weight.copy_(mha.out_proj.weight)
            combined.out_proj.bias.copy_(mha.out_proj.bias)
        x = torch.randn(2, 6, 16)
        self.assertTrue(torch.allclose(mha(x), combined(x), atol=1e-5))


class TestQwen3OptimizedCache(unittest.TestCase):
    """bugs.md #2/#3: optimized 注意力的 RoPE 张量与 layer_idx 条件。"""

    def test_forward_with_cache_runs(self):
        torch.manual_seed(0)
        attn = GroupedQueryAttention_Qwen3_optimized(
            d_model=32, n_heads=4, num_kv_groups=2, qk_norm=True,
        ).eval()
        from layers.position_encoding.RoPE import precompute_rope_params
        cos, sin = precompute_rope_params(head_dim=8, context_length=16)
        cache = KVCache_optimized(
            n_layers=1, max_len=16, num_kv_groups=2, head_dim=8,
            device=torch.device("cpu"), dtype=torch.float32,
        )
        x = torch.randn(1, 5, 32)
        mask = torch.triu(torch.ones(5, 5, dtype=torch.bool), diagonal=1)
        out = attn(x, mask, cos, sin, start_pos=0, cache=cache, layer_idx=0)
        self.assertEqual(out.shape, (1, 5, 32))
        # 增量一步：cache 长度应变为 6
        x1 = torch.randn(1, 1, 32)
        mask1 = torch.zeros(1, 6, dtype=torch.bool)
        out1 = attn(x1, mask1, cos, sin, start_pos=5, cache=cache, layer_idx=0)
        self.assertEqual(out1.shape, (1, 1, 32))

    def test_missing_layer_idx_raises(self):
        attn = GroupedQueryAttention_Qwen3_optimized(d_model=32, n_heads=4, num_kv_groups=2)
        from layers.position_encoding.RoPE import precompute_rope_params
        cos, sin = precompute_rope_params(head_dim=8, context_length=16)
        cache = KVCache_optimized(1, 16, 2, 8, torch.device("cpu"), torch.float32)
        x = torch.randn(1, 3, 32)
        mask = torch.triu(torch.ones(3, 3, dtype=torch.bool), diagonal=1)
        with self.assertRaises(ValueError):
            attn(x, mask, cos, sin, cache=cache)  # layer_idx=None 必须报错


class TestGPT2CacheAndLoaderCompat(unittest.TestCase):
    """bugs.md #8: GPT-2 的 reset_kv_cache / use_cache 必须真正工作；
    属性命名必须与 inference/loaders/gpt2 的加载约定一致。"""

    def setUp(self):
        torch.manual_seed(123)
        self.cfg = _gpt2_cfg()
        self.model = GPT2Model(self.cfg).eval()

    def test_reset_kv_cache_no_crash(self):
        self.model.reset_kv_cache()

    def test_loader_attribute_names(self):
        blk = self.model.trf_blocks[0]
        self.assertTrue(hasattr(blk.attn, "W_query"))
        self.assertTrue(hasattr(blk.ff, "layers"))
        self.assertTrue(hasattr(blk.norm1, "scale"))
        self.assertTrue(hasattr(self.model.final_norm, "scale"))

    def test_cache_matches_no_cache_greedy(self):
        """同一模型开/关 KV cache 的 greedy 生成必须逐 token 一致。"""
        x = torch.randint(0, self.cfg.vocab_size, (1, 5))
        with torch.no_grad():
            # no cache
            ids_nc = x.clone()
            for _ in range(4):
                logits = self.model(ids_nc)
                ids_nc = torch.cat([ids_nc, logits[:, -1].argmax(dim=-1, keepdim=True)], dim=1)
            # with cache
            self.model.reset_kv_cache()
            ids_c = x.clone()
            logits = self.model(ids_c, use_cache=True)
            for _ in range(4):
                nxt = logits[:, -1].argmax(dim=-1, keepdim=True)
                ids_c = torch.cat([ids_c, nxt], dim=1)
                logits = self.model(nxt, use_cache=True)
        self.assertTrue(torch.equal(ids_nc, ids_c))


class TestEmbeddingAndPE(unittest.TestCase):
    """bugs.md #6/#7: embedding assert 与 FixPE 重复类。"""

    def test_absolute_position_embeddings(self):
        pe = AbsolutePositionEmbeddings(d_model=16, context_length=8)
        out = pe(torch.arange(4))
        self.assertEqual(out.shape, (4, 16))

    def test_fixpe_forward(self):
        pe = PositionalEncoding(d_model=8, max_len=16)
        out = pe(torch.zeros(2, 5, 8))
        self.assertEqual(out.shape, (2, 5, 8))

    def test_fixpe_with_dropout(self):
        cfg = SimpleNamespace(dropout=0.0, block_size=16, embed_dim=8)
        pe = PositionalEncodingWithDropout(cfg)
        out = pe(torch.zeros(2, 5, 8))
        self.assertEqual(out.shape, (2, 5, 8))


class TestFeedForward(unittest.TestCase):
    """bugs.md P0-9 关联：d_ff/hidden_dim 回退；SwiGLU 不再是空壳。"""

    def test_silu_hidden_dim_fallback(self):
        cfg = SimpleNamespace(embed_dim=16, hidden_dim=32, dtype=torch.float32)
        ff = FeedForwardSiLU(cfg)
        out = ff(torch.randn(2, 3, 16))
        self.assertEqual(out.shape, (2, 3, 16))

    def test_swiglu_not_empty(self):
        cfg = SimpleNamespace(embed_dim=16, d_ff=32, dtype=torch.float32)
        ff = FeedForwardSwiGLU(cfg)
        out = ff(torch.randn(2, 3, 16))
        self.assertIsNotNone(out)
        self.assertEqual(out.shape, (2, 3, 16))

    def test_activation_swiglu(self):
        act = SwiGLU()
        x = torch.randn(2, 8)
        out = act(x)
        self.assertEqual(out.shape, (2, 4))
        x1, x2 = x.chunk(2, dim=-1)
        self.assertTrue(torch.allclose(out, torch.nn.functional.silu(x1) * x2))

    def test_gelu_sequential_loader_compat(self):
        ff = FeedForwardGELU(SimpleNamespace(embed_dim=16, d_ff=32, dtype=torch.float32))
        self.assertIsInstance(ff.layers[0], torch.nn.Linear)
        self.assertIsInstance(ff.layers[2], torch.nn.Linear)


class TestRMSNormNumerics(unittest.TestCase):
    """bugs.md #14: 基础 RMSNorm 与 PyTorch 内置实现一致（fp32 上转后）。"""

    def test_matches_torch_rmsnorm(self):
        torch.manual_seed(0)
        ours = RMSNorm(embed_dim=32, eps=1e-6)
        ref = torch.nn.RMSNorm(32, eps=1e-6)
        x = torch.randn(4, 7, 32)
        self.assertTrue(torch.allclose(ours(x), ref(x), atol=1e-5))

    def test_bf16_input(self):
        ours = RMSNorm(embed_dim=32).eval()
        x = torch.randn(2, 5, 32, dtype=torch.bfloat16)
        out = ours(x)
        self.assertEqual(out.dtype, torch.bfloat16)
        self.assertTrue(torch.isfinite(out).all())


class TestSimpleBPERoundtrip(unittest.TestCase):
    """bugs.md #16: 自训练 BPE 的 encode/decode 必须互逆，且按 rank 合并。"""

    def test_train_encode_decode_roundtrip(self):
        from layers.tokenizers.bpe.simple_bpe import SimpleBPE
        corpus = "low lower lowest low lower low " * 20
        tok = SimpleBPE(
            train=True, load_openai=False, name="test_bpe_roundtrip",
            corpus=corpus, vocab_size=300,
        )
        text = "low lowest lower low"
        ids = tok.encode(text)
        self.assertGreater(tok.n_vocab, 256)  # 真的学到了 merge
        self.assertEqual(tok.decode(ids), text)


class TestQwen3BookStateDictRemap(unittest.TestCase):
    """Phase 1: qwen3_original 与 qwen3 合并后，书本 checkpoint 键名可无损重映射。"""

    def test_remap_is_identity_on_canonical_and_roundtrip(self):
        from models.qwen3.qwen3 import Model as Qwen3Model
        from models.io.qwen3.load_weights import remap_book_state_dict
        cfg = SimpleNamespace(
            vocab_size=64, context_length=16, embed_dim=32, n_heads=4,
            n_layers=2, head_dim=8, qk_norm=True, n_kv_groups=2,
            rope_base=1_000_000.0, dtype=torch.float32,
            num_experts=0, num_experts_per_tok=0, moe_intermediate_size=0,
            hidden_dim=64,
        )
        model = Qwen3Model(cfg)
        sd = model.state_dict()
        # 规范命名不应被改动
        self.assertEqual(set(remap_book_state_dict(sd).keys()), set(sd.keys()))
        # 模拟书本命名（tok_emb / .att.）再重映射回来，必须还原
        book = {k.replace("tok_embed.", "tok_emb.").replace(".attn.", ".att."): v
                for k, v in sd.items()}
        remapped = remap_book_state_dict(book)
        self.assertEqual(set(remapped.keys()), set(sd.keys()))
        model.load_state_dict(remapped)  # strict 加载通过


if __name__ == "__main__":
    unittest.main()
