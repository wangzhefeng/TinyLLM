import unittest
from types import SimpleNamespace

import torch

from config.base import GenerationConfig, ModelConfig, TrainConfig
from layers.kv_cache import KVCache
from models.gpt2.gpt2_124M import Model as GPT2Model
from models.qwen3.qwen3 import Model as Qwen3Model


def gpt_config():
    return SimpleNamespace(
        vocab_size=64, context_length=32, embed_dim=16, d_ff=32,
        n_heads=4, n_layers=2, dropout=0.0, qkv_bias=True,
        dtype=torch.float32,
    )


def qwen_config():
    return SimpleNamespace(
        vocab_size=64, context_length=32, embed_dim=16, d_ff=32,
        n_heads=4, n_layers=2, n_kv_groups=2, head_dim=4,
        dropout=0.0, qk_norm=True, rope_base=10000.0,
        num_experts=0, dtype=torch.float32,
    )


class TestExternalCacheContract(unittest.TestCase):
    def assert_incremental_and_isolated(self, model):
        model.eval()
        prompt = torch.tensor([[1, 2, 3]])
        other = torch.tensor([[4, 5]])
        next_token = torch.tensor([[6]])
        cache_a = KVCache(2)
        cache_b = KVCache(2)
        with torch.inference_mode():
            model(prompt, cache=cache_a)
            model(other, cache=cache_b)
            incremental = model(next_token, cache=cache_a)
            full = model(torch.cat((prompt, next_token), dim=1))
        self.assertEqual(cache_a.position, 4)
        self.assertEqual(cache_b.position, 2)
        self.assertTrue(torch.allclose(incremental[:, -1], full[:, -1], atol=1e-5, rtol=1e-5))

    def test_gpt2_external_cache(self):
        self.assert_incremental_and_isolated(GPT2Model(gpt_config()))

    def test_qwen3_external_cache(self):
        self.assert_incremental_and_isolated(Qwen3Model(qwen_config()))

    def test_legacy_gpt2_cache_still_works(self):
        model = GPT2Model(gpt_config()).eval()
        model.reset_kv_cache()
        with torch.inference_mode():
            first = model(torch.tensor([[1, 2]]), True)
            second = model(torch.tensor([[3]]), True)
            full = model(torch.tensor([[1, 2, 3]]))
        self.assertTrue(torch.allclose(second[:, -1], full[:, -1], atol=1e-5, rtol=1e-5))


class TestConfigContract(unittest.TestCase):
    def test_model_config_dict_and_hidden_dim_compatibility(self):
        cfg = ModelConfig.from_dict({
            "vocab_size": 100, "context_length": 32, "embed_dim": 16,
            "n_heads": 4, "n_layers": 2, "hidden_dim": 48,
        })
        self.assertEqual(cfg.d_ff, 48)
        self.assertEqual(cfg.hidden_dim, 48)
        self.assertEqual(cfg["vocab_size"], 100)
        self.assertEqual(cfg.copy().to_dict(), cfg.to_dict())

    def test_train_and_generation_configs_are_separate(self):
        self.assertEqual(TrainConfig().grad_accum_steps, 1)
        self.assertTrue(GenerationConfig().use_cache)
        self.assertNotIn("temperature", ModelConfig.from_dict({
            "vocab_size": 1, "context_length": 1, "embed_dim": 1,
            "n_heads": 1, "n_layers": 1, "temperature": 0.5,
        }).to_dict())

    def test_canonical_qwen_and_llama_configs(self):
        from config.llama.model_cfgs import get_config
        from config.qwen3.model_cfgs import get_cfgs
        self.assertEqual(get_cfgs("0.6B").n_layers, 28)
        self.assertEqual(get_config("llama2").d_ff, 11008)


class TestTokenizerRegistry(unittest.TestCase):
    def test_case_insensitive_gpt2_alias(self):
        from layers.tokenizers.tokenization import choose_tokenizer
        tokenizer = choose_tokenizer("GPT2")
        self.assertGreater(tokenizer.n_vocab, 50_000)
        self.assertIsInstance(tokenizer.eos_token_id, int)

    def test_unknown_tokenizer_lists_available(self):
        from layers.tokenizers.tokenization import choose_tokenizer
        with self.assertRaisesRegex(ValueError, "qwen3"):
            choose_tokenizer("not-a-tokenizer")


if __name__ == "__main__":
    unittest.main()
