"""Olmo 3 tiny 模型测试：forward 形状/有限性、backward 梯度、参数量粗检、
滑窗行为与 KV cache 增量一致性。离线 CPU 运行。"""
import unittest

import torch

from config.olmo3.model_cfgs import get_cfgs
from layers.position_encoding.RoPE import precompute_rope_params
from models.base import CausalLM
from models.olmo3.olmo3 import GroupedQueryAttention_Olmo3, Model


class TestOlmo3Config(unittest.TestCase):
    def test_7b_official_defaults(self):
        # transformers Olmo3Config 类默认值（checkpoint 级 config.json 为 gated，未能核实）
        cfg = get_cfgs("7B")
        self.assertEqual(cfg.vocab_size, 50_304)
        self.assertEqual(cfg.embed_dim, 4_096)
        self.assertEqual(cfg.n_layers, 32)
        self.assertEqual(cfg.n_heads, 32)
        self.assertEqual(cfg.n_kv_groups, 32)
        self.assertEqual(cfg.d_ff, 11_008)
        self.assertEqual(cfg.rope_base, 500_000.0)
        self.assertEqual(cfg.sliding_window, 4_096)
        self.assertEqual(cfg.full_attention_interval, 4)

    def test_tiny_config(self):
        cfg = get_cfgs("TINY")
        self.assertEqual(cfg.n_layers, 4)
        self.assertEqual(cfg.tie_word_embeddings, True)

    def test_unknown_model_raises(self):
        with self.assertRaises(ValueError):
            get_cfgs("99B")


class TestOlmo3Forward(unittest.TestCase):
    def setUp(self):
        self.cfg = get_cfgs("TINY")
        torch.manual_seed(0)
        self.model = Model(self.cfg).eval()

    def test_layer_pattern(self):
        # 每 4 层 1 层 full attention，其余 3 层 sliding attention
        self.assertEqual(
            self.model.layer_types,
            ["sliding_attention"] * 3 + ["full_attention"],
        )

    def test_forward_shape_and_finiteness(self):
        x = torch.randint(0, self.cfg.vocab_size, (2, 16))
        with torch.no_grad():
            logits = self.model(x)
        self.assertEqual(logits.shape, (2, 16, self.cfg.vocab_size))
        self.assertTrue(torch.isfinite(logits).all())

    def test_backward_grads_finite(self):
        x = torch.randint(0, self.cfg.vocab_size, (1, 8))
        logits = self.model(x)
        logits.float().sum().backward()
        grads = [p.grad for p in self.model.parameters() if p.grad is not None]
        self.assertGreater(len(grads), 0)
        for grad in grads:
            self.assertTrue(torch.isfinite(grad).all())

    def test_param_count(self):
        # 全程无 bias；QK-norm 非参数化（不贡献参数）
        cfg = self.cfg
        D, d_ff, L, V = cfg.embed_dim, cfg.d_ff, cfg.n_layers, cfg.vocab_size
        n_h, n_kv, hd = cfg.n_heads, cfg.n_kv_groups, cfg.head_dim
        per_layer = D * hd * 2 * (n_h + n_kv) + 3 * D * d_ff + 2 * D
        expected = V * D + L * per_layer + D  # 权重共享：out_head 不计
        self.assertEqual(sum(p.numel() for p in self.model.parameters()), expected)

    def test_qk_norm_has_no_parameters(self):
        # Olmo 风格 QK-norm 为非参数化 RMSNorm
        attn = self.model.trf_blocks[0].attn
        self.assertEqual(list(attn.q_norm.parameters()), [])
        self.assertEqual(list(attn.k_norm.parameters()), [])

    def test_satisfies_causallm_protocol(self):
        self.assertIsInstance(self.model, CausalLM)


class TestOlmo3SlidingWindow(unittest.TestCase):
    def test_out_of_window_tokens_do_not_matter(self):
        # 窗口外的历史 token 不影响 sliding 层输出（对 attention 模块直接验证）
        torch.manual_seed(0)
        window = 4
        attn = GroupedQueryAttention_Olmo3(
            d_model = 32, n_heads = 4, num_kv_groups = 2, head_dim = 8,
        ).eval()
        cos, sin = precompute_rope_params(
            head_dim = 8, theta_base = 500_000.0, context_length = 32,
        )
        num_tokens = 12
        pos = torch.arange(num_tokens)
        dist = pos.view(-1, 1) - pos.view(1, -1)
        swa_mask = ((dist < 0) | (dist >= window))[None, None, :, :]
        x1 = torch.randn(1, num_tokens, 32)
        x2 = x1.clone()
        x2[:, 0, :] = torch.randn(1, 32)
        with torch.no_grad():
            out1, _ = attn(x1, swa_mask, cos, sin)
            out2, _ = attn(x2, swa_mask, cos, sin)
        # 位置 0 只被窗口内（0..window-1）token 看到
        torch.testing.assert_close(out1[:, window:, :], out2[:, window:, :])


class TestOlmo3KVCache(unittest.TestCase):
    def test_incremental_matches_full(self):
        # 序列长度超过 sliding_window(6)，覆盖滑窗 mask 的 cache 偏移正确性
        torch.manual_seed(0)
        model = Model(get_cfgs("TINY")).eval()
        prompt = torch.randint(0, 64, (1, 10))
        next_token = torch.randint(0, 64, (1, 1))
        with torch.inference_mode():
            cache = model.new_cache()
            model(prompt, cache = cache)
            incremental = model(next_token, cache = cache)
            full = model(torch.cat((prompt, next_token), dim = 1))
        self.assertEqual(cache.position, 11)
        torch.testing.assert_close(
            incremental[:, -1], full[:, -1], atol = 1e-5, rtol = 1e-5,
        )

    def test_caches_are_isolated(self):
        torch.manual_seed(0)
        model = Model(get_cfgs("TINY")).eval()
        cache_a = model.new_cache()
        cache_b = model.new_cache()
        with torch.inference_mode():
            model(torch.randint(0, 64, (1, 5)), cache = cache_a)
            model(torch.randint(0, 64, (1, 3)), cache = cache_b)
        self.assertEqual(cache_a.position, 5)
        self.assertEqual(cache_b.position, 3)


if __name__ == "__main__":
    unittest.main()
