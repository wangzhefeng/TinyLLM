"""Weight tying（out_head 与 tok_embed 共享）测试。"""
import unittest
from types import SimpleNamespace

import torch

from models.gpt2.gpt2_124M import Model as GPT2Model
from models.llama.llama3_8B import Model as Llama3Model


def _gpt2_cfg(**overrides):
    cfg = dict(
        vocab_size=128,
        context_length=16,
        embed_dim=32,
        d_ff=64,
        n_heads=4,
        n_layers=2,
        dropout=0.0,
        qkv_bias=True,
        dtype=torch.float32,
        is_train=False,
        tie_word_embeddings=False,
    )
    cfg.update(overrides)
    return SimpleNamespace(**cfg)


def _llama_cfg(**overrides):
    cfg = dict(
        vocab_size=128,
        context_length=16,
        embed_dim=32,
        n_heads=4,
        n_kv_groups=2,
        n_layers=2,
        hidden_dim=64,
        rope_base=10_000.0,
        rope_freq=None,
        dtype=torch.float32,
        tie_word_embeddings=False,
    )
    cfg.update(overrides)
    return SimpleNamespace(**cfg)


class TestGPT2WeightTying(unittest.TestCase):
    def test_tied_weights_share_parameter(self):
        cfg = _gpt2_cfg(tie_word_embeddings=True)
        model = GPT2Model(cfg)
        self.assertIs(model.out_head.weight, model.tok_embed.weight)

    def test_untied_by_default(self):
        model = GPT2Model(_gpt2_cfg())
        self.assertIsNot(model.out_head.weight, model.tok_embed.weight)

    def test_param_count_reduced(self):
        cfg = _gpt2_cfg(tie_word_embeddings=True)
        tied = GPT2Model(cfg)
        untied = GPT2Model(_gpt2_cfg(tie_word_embeddings=False))
        tied_params = sum(p.numel() for p in tied.parameters())
        untied_params = sum(p.numel() for p in untied.parameters())
        self.assertEqual(untied_params - tied_params, cfg.vocab_size * cfg.embed_dim)

    def test_forward_with_tying(self):
        torch.manual_seed(123)
        model = GPT2Model(_gpt2_cfg(tie_word_embeddings=True)).eval()
        x = torch.randint(0, 128, (2, 8))
        with torch.no_grad():
            logits = model(x)
        self.assertEqual(logits.shape, (2, 8, 128))


class TestLlama3WeightTying(unittest.TestCase):
    def test_tied_weights_share_parameter(self):
        cfg = _llama_cfg(tie_word_embeddings=True)
        model = Llama3Model(cfg)
        self.assertIs(model.out_head.weight, model.tok_embed.weight)
        untied = Llama3Model(_llama_cfg(tie_word_embeddings=False))
        self.assertIsNot(untied.out_head.weight, untied.tok_embed.weight)

    def test_forward_with_tying(self):
        torch.manual_seed(123)
        model = Llama3Model(_llama_cfg(tie_word_embeddings=True)).eval()
        x = torch.randint(0, 128, (2, 8))
        with torch.no_grad():
            logits = model(x)
        self.assertEqual(logits.shape, (2, 8, 128))


if __name__ == "__main__":
    unittest.main()
