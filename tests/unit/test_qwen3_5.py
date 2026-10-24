"""Qwen3.5 tiny 模型测试：forward 形状/有限性、backward 梯度、参数量粗检、
gated delta net 因果性与分块一致性、KV cache 增量一致性、配置注册。离线 CPU 运行。"""
import unittest
from types import SimpleNamespace

import torch

from config.qwen3.model_cfgs import get_cfgs
from models.base import CausalLM
from models.qwen3.qwen3_5 import GatedDeltaNet_Qwen3_5, Model


def tiny_cfg():
    # 结构对照 Qwen3.5-4B 等比缩小：4 层 = 3 层 linear attention + 第 4 层 full attention
    return SimpleNamespace(
        vocab_size = 64, context_length = 32, embed_dim = 32, d_ff = 64,
        n_heads = 4, n_layers = 4, n_kv_groups = 2, head_dim = 16,
        qk_norm = True, rope_base = 10_000_000.0,
        num_experts = 0, dtype = torch.float32,
        full_attention_interval = 4, attn_output_gate = True,
        partial_rotary_factor = 0.5,
        linear_num_key_heads = 2, linear_key_head_dim = 8,
        linear_num_value_heads = 4, linear_value_head_dim = 8,
        linear_conv_kernel_dim = 4,
        tie_word_embeddings = True,
    )


class TestQwen3_5Forward(unittest.TestCase):
    def setUp(self):
        self.cfg = tiny_cfg()
        torch.manual_seed(0)
        self.model = Model(self.cfg).eval()

    def test_layer_pattern(self):
        # 每 4 层 1 层 full attention，其余 3 层 linear attention
        self.assertEqual(
            self.model.layer_types,
            ["linear_attention"] * 3 + ["full_attention"],
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
        # gated delta net 特有参数收到梯度
        linear_attn = self.model.trf_blocks[0].attn
        self.assertIsNotNone(linear_attn.A_log.grad)
        self.assertIsNotNone(linear_attn.dt_bias.grad)

    def test_param_count(self):
        # 全程无 bias；权重共享（out_head 不计）
        cfg = self.cfg
        D, d_ff, V = cfg.embed_dim, cfg.d_ff, cfg.vocab_size
        n_h, n_kv, hd = cfg.n_heads, cfg.n_kv_groups, cfg.head_dim
        n_k_h, k_hd = cfg.linear_num_key_heads, cfg.linear_key_head_dim
        n_v_h, v_hd = cfg.linear_num_value_heads, cfg.linear_value_head_dim
        key_dim, value_dim = n_k_h * k_hd, n_v_h * v_hd
        conv_dim = 2 * key_dim + value_dim
        ffn = 3 * D * d_ff + 2 * D
        full_layer = D * n_h * hd * 2 + 2 * D * n_kv * hd + n_h * hd * D + 2 * hd + ffn
        linear_layer = (
            D * conv_dim + D * value_dim + 2 * D * n_v_h     # qkv/z/ba 投影
            + conv_dim * cfg.linear_conv_kernel_dim           # depthwise conv1d
            + 2 * n_v_h + v_hd                                 # dt_bias/A_log + 门控 norm
            + value_dim * D                                     # out_proj
            + ffn
        )
        expected = V * D + 3 * linear_layer + full_layer + D  # final norm
        self.assertEqual(sum(p.numel() for p in self.model.parameters()), expected)

    def test_weight_tying(self):
        self.assertIs(self.model.out_head.weight, self.model.tok_embed.weight)

    def test_partial_rope_dim(self):
        # rotary_dim = head_dim * partial_rotary_factor = 16 * 0.5 = 8
        self.assertEqual(self.model.rotary_dim, 8)
        self.assertEqual(self.model.cos.shape[-1], 8)

    def test_satisfies_causallm_protocol(self):
        self.assertIsInstance(self.model, CausalLM)


class TestGatedDeltaNet(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.net = GatedDeltaNet_Qwen3_5(
            d_model = 32, num_k_heads = 2, head_k_dim = 8,
            num_v_heads = 4, head_v_dim = 8,
        ).eval()

    def test_future_tokens_do_not_affect_past(self):
        # 因果性：只篡改未来 token，过去位置的输出不变（conv 与递推均只向左看）
        x1 = torch.randn(1, 8, 32)
        x2 = x1.clone()
        x2[:, 5:, :] = torch.randn(1, 3, 32)
        with torch.no_grad():
            out1, _ = self.net(x1)
            out2, _ = self.net(x2)
        torch.testing.assert_close(out1[:, :5, :], out2[:, :5, :])

    def test_chunked_matches_full(self):
        # 分块递推（携带 conv/recurrent 状态）与整段前向一致
        x = torch.randn(1, 10, 32)
        with torch.no_grad():
            full, _ = self.net(x)
            out1, cache = self.net(x[:, :6], cache = None)
            out2, _ = self.net(x[:, 6:], cache = cache)
        torch.testing.assert_close(
            torch.cat([out1, out2], dim = 1), full, atol = 1e-5, rtol = 1e-5,
        )

    def test_cache_state_shapes(self):
        # 推断状态：conv (B, conv_dim, k-1)；recurrent (B, num_v_heads, K, V)
        with torch.no_grad():
            _, (conv_state, rec_state) = self.net(torch.randn(1, 4, 32))
        self.assertEqual(conv_state.shape, (1, 64, 3))
        self.assertEqual(rec_state.shape, (1, 4, 8, 8))


class TestQwen3_5KVCache(unittest.TestCase):
    def test_incremental_matches_full(self):
        # 覆盖 linear 层（conv+递推状态）与 full 层（KV cache）的增量一致性
        torch.manual_seed(0)
        model = Model(tiny_cfg()).eval()
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
        model = Model(tiny_cfg()).eval()
        cache_a = model.new_cache()
        cache_b = model.new_cache()
        with torch.inference_mode():
            model(torch.randint(0, 64, (1, 5)), cache = cache_a)
            model(torch.randint(0, 64, (1, 3)), cache = cache_b)
        self.assertEqual(cache_a.position, 5)
        self.assertEqual(cache_b.position, 3)


class TestQwen3_5ConfigRegistration(unittest.TestCase):
    def test_official_4b_config(self):
        # 官方 config.json（Qwen/Qwen3.5-4B，text_config）关键超参
        cfg = get_cfgs("Qwen3.5-4B")
        self.assertEqual(cfg.vocab_size, 248_320)
        self.assertEqual(cfg.embed_dim, 2_560)
        self.assertEqual(cfg.n_layers, 32)
        self.assertEqual(cfg.n_heads, 16)
        self.assertEqual(cfg.n_kv_groups, 4)
        self.assertEqual(cfg.head_dim, 256)
        self.assertEqual(cfg.d_ff, 9_216)
        self.assertEqual(cfg.rope_base, 10_000_000.0)
        self.assertEqual(cfg.full_attention_interval, 4)
        self.assertEqual(cfg.attn_output_gate, True)
        self.assertEqual(cfg.partial_rotary_factor, 0.25)
        self.assertEqual(cfg.linear_num_key_heads, 16)
        self.assertEqual(cfg.linear_key_head_dim, 128)
        self.assertEqual(cfg.linear_num_value_heads, 32)
        self.assertEqual(cfg.linear_value_head_dim, 128)
        self.assertEqual(cfg.linear_conv_kernel_dim, 4)
        self.assertEqual(cfg.tie_word_embeddings, True)

    def test_existing_qwen3_entries_untouched(self):
        self.assertEqual(get_cfgs("0.6B").n_layers, 28)
        self.assertEqual(get_cfgs("0.6B").vocab_size, 151_936)
        self.assertEqual(get_cfgs("30B-A3B").num_experts, 128)


if __name__ == "__main__":
    unittest.main()
