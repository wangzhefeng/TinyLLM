import unittest
from types import SimpleNamespace

import torch

from inference.generate_core import generate, generate_qwen3, sample_next_token
from models.gpt2.gpt2_124M import Model


class TestSampleNextToken(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.logits = torch.randn(1, 50) * 5  # contains negative values

    def test_greedy_matches_argmax(self):
        logits = torch.randn(2, 100)
        out = sample_next_token(logits, temperature=0.0)
        self.assertTrue(torch.equal(out, logits.argmax(dim=-1, keepdim=True)))

    def test_temperature_sampling_with_negative_logits(self):
        # regression: multinomial must receive softmax probs, not raw logits
        out = sample_next_token(self.logits, temperature=1.0)
        self.assertEqual(out.shape, (1, 1))

    def test_top_k_one_equals_greedy(self):
        out = sample_next_token(self.logits, temperature=0.7, top_k=1)
        self.assertTrue(torch.equal(out, self.logits.argmax(dim=-1, keepdim=True)))

    def test_top_k_batched_no_broadcast_error(self):
        # regression: min_val must keep dim for batched logits
        logits = torch.randn(4, 30)
        out = sample_next_token(logits, temperature=1.2, top_k=5, top_p=0.95)
        self.assertEqual(out.shape, (4, 1))

    def test_top_p_keeps_top_token_when_tiny(self):
        out = sample_next_token(self.logits, temperature=1.0, top_p=0.01)
        self.assertTrue(torch.equal(out, self.logits.argmax(dim=-1, keepdim=True)))

    def test_reproducible_with_seed(self):
        torch.manual_seed(42)
        a = sample_next_token(self.logits, 1.0, top_k=20, top_p=0.9)
        torch.manual_seed(42)
        b = sample_next_token(self.logits, 1.0, top_k=20, top_p=0.9)
        self.assertTrue(torch.equal(a, b))


class TestGenerate(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)
        self.cfg = SimpleNamespace(
            vocab_size=128,
            context_length=32,
            embed_dim=32,
            d_ff=64,
            n_heads=4,
            n_layers=2,
            dropout=0.0,
            qkv_bias=True,
            dtype=torch.float32,
            is_train=False,
        )
        self.model = Model(self.cfg).eval()
        self.prompt = torch.randint(0, self.cfg.vocab_size, (1, 5))

    def test_greedy_shape_and_determinism(self):
        out1 = generate(self.model, self.prompt.clone(), max_new_tokens=6,
                        context_length=self.cfg.context_length)
        out2 = generate(self.model, self.prompt.clone(), max_new_tokens=6,
                        context_length=self.cfg.context_length)
        self.assertEqual(out1.shape, (1, 11))
        self.assertTrue(torch.equal(out1, out2))

    def test_temperature_top_k_top_p_runs(self):
        torch.manual_seed(7)
        out = generate(self.model, self.prompt.clone(), max_new_tokens=6,
                       context_length=self.cfg.context_length,
                       temperature=1.0, top_k=10, top_p=0.9)
        self.assertEqual(out.shape, (1, 11))

    def test_kv_cache_matches_no_cache_greedy(self):
        # GPT-2 与 Qwen3 均已实现 KV cache；此处用 Qwen3 验证 cached/uncached 一致性
        from models.qwen3.qwen3 import Model as Qwen3Model
        cfg = SimpleNamespace(
            vocab_size=128, context_length=32, embed_dim=32, n_heads=4,
            n_kv_groups=2, head_dim=8, n_layers=2, dtype=torch.bfloat16,
            qk_norm=True, qkv_bias=False, rope_base=10000, rope_freq=None,
            num_experts=0, query_pre_attn_scalar=None, kv_window_size=32,
            dropout=0.0, is_train=False, d_ff=64,
        )
        torch.manual_seed(0)
        model = Qwen3Model(cfg).eval()
        prompt = torch.randint(0, cfg.vocab_size, (1, 5))
        out_plain = generate_qwen3(model, prompt.clone(), max_new_tokens=6,
                                   context_length=32, use_cache=False)
        out_cache = generate_qwen3(model, prompt.clone(), max_new_tokens=6,
                                   context_length=32, use_cache=True)
        self.assertEqual(out_plain.shape, (1, 6))
        self.assertTrue(torch.equal(out_plain, out_cache))

    def test_gpt2_kv_cache_matches_no_cache_greedy(self):
        # GPT-2（models/gpt2/gpt2_124M.py）已实现 KV cache：
        # cached 与 uncached 贪心生成必须逐 token 一致
        out_plain = generate(self.model, self.prompt.clone(), max_new_tokens=6,
                             context_length=self.cfg.context_length, use_cache=False)
        out_cache = generate(self.model, self.prompt.clone(), max_new_tokens=6,
                             context_length=self.cfg.context_length, use_cache=True)
        self.assertEqual(out_cache.shape, (1, 11))
        self.assertTrue(torch.equal(out_plain, out_cache))

    def test_eos_early_stop(self):
        # 用全正隐藏状态确定性地强制 EOS，避免不同后端下随机隐藏状态改变 argmax
        eos_id = 3
        with torch.no_grad():
            for parameter in self.model.parameters():
                parameter.zero_()
            self.model.tok_embed.weight.fill_(1.0)
            self.model.pos_embed.weight.fill_(1.0)
            self.model.final_norm.scale.fill_(1.0)
            self.model.final_norm.shift.fill_(1.0)
            self.model.out_head.weight[eos_id, :].fill_(1.0)
        out = generate(self.model, self.prompt.clone(), max_new_tokens=10,
                       context_length=self.cfg.context_length, eos_token_id=eos_id)
        # stops right after the first generated token
        self.assertEqual(out.shape, (1, 5))
        self.assertLessEqual(out.shape[1], 6)


class TestBatchedGenerateQwen3(unittest.TestCase):
    """批量右填充生成与逐条生成（贪心）逐 token 一致——掩码/取位语义契约。"""

    def setUp(self):
        from models.qwen3.qwen3 import Model as Qwen3Model
        torch.manual_seed(0)
        self.cfg = SimpleNamespace(
            vocab_size=128, context_length=32, embed_dim=32, n_heads=4,
            n_kv_groups=2, head_dim=8, n_layers=2, dtype=torch.bfloat16,
            qk_norm=True, qkv_bias=False, rope_base=10000, rope_freq=None,
            num_experts=0, query_pre_attn_scalar=None, kv_window_size=32,
            dropout=0.0, is_train=False, d_ff=64,
        )
        self.model = Qwen3Model(self.cfg).eval()

    def test_batched_padding_matches_serial_greedy(self):
        from inference.generate_core import (
            generate_qwen3,
            generate_qwen3_batched_cache,
        )
        prompt_a = torch.randint(0, self.cfg.vocab_size, (1, 5))
        prompt_b = torch.randint(0, self.cfg.vocab_size, (1, 8))
        pad_id = 0
        token_ids = torch.full((2, 8), pad_id, dtype=torch.long)
        token_ids[0, :5] = prompt_a[0]
        token_ids[1] = prompt_b[0]
        attn_mask = torch.tensor(
            [[1] * 5 + [0] * 3, [1] * 8], dtype=torch.bool,
        )
        batched = generate_qwen3_batched_cache(
            self.model, token_ids, max_new_tokens=6, context_length=32,
            eos_token_id=None, attn_mask=attn_mask, pad_id=pad_id,
        )
        for prompt, row in ((prompt_a, batched[0]), (prompt_b, batched[1])):
            serial = generate_qwen3(
                self.model, prompt.clone(), max_new_tokens=6,
                context_length=32, use_cache=True,
            )
            self.assertTrue(
                torch.equal(serial[0], row),
                f"batched row diverged from serial: {serial[0].tolist()} vs {row.tolist()}",
            )


if __name__ == "__main__":
    unittest.main()
