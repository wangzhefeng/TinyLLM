"""可解释性工具测试：trace 与 generate 的行为一致性、attention 捕获契约。"""
import unittest
from types import SimpleNamespace

import torch

from inference.generate_core import generate
from inference.interpretability import (
    attention_weights,
    render_attention_md,
    render_trace_md,
    trace_generation,
)
from layers.attention import MultiHeadAttention
from models.gpt2.gpt2_124M import Model

VOCAB, CTX, EMB, D_FF, N_LAYERS = 128, 16, 32, 64, 2


def _tiny_cfg():
    return SimpleNamespace(
        vocab_size=VOCAB, context_length=CTX, embed_dim=EMB, d_ff=D_FF,
        n_heads=4, n_layers=N_LAYERS, dropout=0.0, qkv_bias=True,
        dtype=torch.float32, is_train=False,
    )


def _tiny_model(seed=0):
    torch.manual_seed(seed)
    return Model(_tiny_cfg()).eval()


def _ids(seed=1, n=6):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, VOCAB, (1, n), generator=g)


class TestTraceGeneration(unittest.TestCase):
    def test_greedy_trace_matches_generate(self):
        """贪心 trace 的 out_ids 必须与正式生成路径逐 token 一致。"""
        model = _tiny_model()
        ids = _ids()
        records, out_ids = trace_generation(model, ids, max_new_tokens=8)
        gen_out = generate(model, ids.clone(), max_new_tokens=8, context_length=None,
                           temperature=0.0)
        self.assertTrue(torch.equal(out_ids, gen_out))
        self.assertEqual(len(records), 8)

    def test_records_topk_and_entropy(self):
        model = _tiny_model()
        records, _ = trace_generation(model, _ids(), max_new_tokens=4, k=5)
        self.assertEqual(len(records), 4)
        for i, rec in enumerate(records):
            self.assertEqual(rec["step"], i)
            self.assertFalse(rec["eos_stop"])
            self.assertIsNone(rec["token"])
            probs = [p for _, _, p in rec["topk"]]
            self.assertEqual(len(probs), 5)
            self.assertEqual(probs, sorted(probs, reverse=True))
            self.assertTrue(all(0.0 <= p <= 1.0 for p in probs))
            self.assertLessEqual(sum(probs), 1.0 + 1e-5)
            self.assertGreaterEqual(rec["entropy"], -1e-6)

    def test_temperature_sampling_runs(self):
        torch.manual_seed(3)
        model = _tiny_model()
        records, out_ids = trace_generation(
            model, _ids(), max_new_tokens=6, temperature=1.0, top_k=10,
        )
        self.assertEqual(len(records), 6)
        self.assertEqual(out_ids.shape, (1, _ids().size(1) + 6))

    def test_eos_stop_matches_generate(self):
        """首步 argmax 恰为 eos：trace 记 1 步且不追加，generate 同样不追加。"""
        model = _tiny_model()
        ids = _ids()
        with torch.no_grad():
            first_greedy = int(torch.argmax(model(ids)[:, -1], dim=-1))
        records, out_ids = trace_generation(
            model, ids, max_new_tokens=5, eos_token_id=first_greedy,
        )
        self.assertEqual(len(records), 1)
        self.assertTrue(records[0]["eos_stop"])
        self.assertTrue(torch.equal(out_ids, ids))
        gen_out = generate(model, ids.clone(), max_new_tokens=5, context_length=None,
                           temperature=0.0, eos_token_id=first_greedy)
        self.assertTrue(torch.equal(gen_out, ids))

    def test_batch_gt1_rejected(self):
        model = _tiny_model()
        with self.assertRaises(ValueError):
            trace_generation(model, torch.randint(0, VOCAB, (2, 4)), max_new_tokens=2)


class TestAttentionCapture(unittest.TestCase):
    def test_model_returns_logits_and_per_layer_weights(self):
        model = _tiny_model()
        ids = _ids()
        logits, attns = attention_weights(model, ids)
        self.assertEqual(logits.shape, (1, ids.size(1), VOCAB))
        self.assertEqual(len(attns), N_LAYERS)
        for w in attns:
            self.assertEqual(tuple(w.shape), (1, 4, ids.size(1), ids.size(1)))

    def test_causal_and_row_sums(self):
        model = _tiny_model()
        ids = _ids()
        _, attns = attention_weights(model, ids)
        T = ids.size(1)
        triu = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
        for w in attns:
            self.assertTrue(torch.all(w[0, :, triu].abs() < 1e-6))
            torch.testing.assert_close(
                w.sum(dim=-1), torch.ones_like(w.sum(dim=-1)), atol=1e-5, rtol=1e-5,
            )

    def test_default_off_unchanged(self):
        model = _tiny_model()
        ids = _ids()
        with torch.no_grad():
            plain = model(ids)
            logits, _ = model(ids, output_attentions=True)
        self.assertIsInstance(plain, torch.Tensor)
        torch.testing.assert_close(plain, logits)

    def test_hidden_states_and_attentions_combined(self):
        model = _tiny_model()
        ids = _ids()
        logits, hidden, attns = model(ids, output_hidden_states=True, output_attentions=True)
        self.assertEqual(hidden.shape, ids.shape + (EMB,))
        self.assertEqual(len(attns), N_LAYERS)

    def test_cache_rejected(self):
        model = _tiny_model()
        ids = _ids()
        with self.assertRaises(ValueError):
            model(ids, use_cache=True, output_attentions=True)
        with self.assertRaises(ValueError):
            model(ids, output_attentions=True, cache=object())

    def test_mha_direct_capture(self):
        torch.manual_seed(0)
        attn = MultiHeadAttention(
            d_model=EMB, d_out=EMB, n_heads=4, context_length=CTX,
            dropout=0.0, qkv_bias=True,
        ).eval()
        x = torch.randn(1, 5, EMB)
        out, w = attn(x, output_attentions=True)
        self.assertEqual(out.shape, (1, 5, EMB))
        self.assertEqual(tuple(w.shape), (1, 4, 5, 5))
        with self.assertRaises(ValueError):
            attn(x, return_cache=True, output_attentions=True)


class TestRenderers(unittest.TestCase):
    def test_render_trace_md(self):
        model = _tiny_model()
        records, _ = trace_generation(model, _ids(), max_new_tokens=3)
        md = render_trace_md(records)
        self.assertIn("| step | token | token_id | entropy |", md)
        # 表头 + 分隔行 + 3 条记录
        self.assertEqual(len([l for l in md.splitlines() if l.startswith("|")]), 5)

    def test_render_attention_md(self):
        w = torch.softmax(torch.randn(1, 4, 3, 3), dim=-1)
        md = render_attention_md(w, tokens=["猫", "吃", "鱼"])
        self.assertIn("猫", md)
        self.assertIn("|---", md)


if __name__ == "__main__":
    unittest.main()
