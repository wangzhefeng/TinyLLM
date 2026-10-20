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
        # GPT-2 attention has no KV cache; use a tiny Qwen3 which implements it
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

    def test_eos_early_stop(self):
        # force EOS as the most likely token by boosting its output logit
        eos_id = 3
        with torch.no_grad():
            self.model.out_head.weight[:, :] = 0.0
            self.model.out_head.weight[eos_id, :] = 1.0
        out = generate(self.model, self.prompt.clone(), max_new_tokens=10,
                       context_length=self.cfg.context_length, eos_token_id=eos_id)
        # stops right after the first generated token
        self.assertEqual(out.shape, (1, 5))
        self.assertLessEqual(out.shape[1], 6)


if __name__ == "__main__":
    unittest.main()
