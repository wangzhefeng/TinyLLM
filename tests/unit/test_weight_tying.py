"""Weight tying（out_head 与 tok_embed 共享）测试。"""
import unittest
from types import SimpleNamespace

import torch

from config.base import ModelConfig
from models.gpt2.gpt2_124M import Model as GPT2Model
from models.llama.llama3_8B import Model as Llama3Model
from models.qwen3.qwen3 import Model as Qwen3Model


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


def _qwen3_cfg(tie):
    return SimpleNamespace(
        vocab_size=64, context_length=16, embed_dim=32, d_ff=64,
        n_heads=4, n_layers=2, n_kv_groups=2, head_dim=8,
        dropout=0.0, qk_norm=True, rope_base=10_000.0, num_experts=0,
        dtype=torch.float32, tie_word_embeddings=tie,
    )


def _qwen3_mock_params(vocab_size=64, embed_dim=32, d_ff=64, n_layers=2):
    params = {
        "model.embed_tokens.weight": torch.randn(vocab_size, embed_dim),
        "model.norm.weight": torch.ones(embed_dim),
    }
    for l in range(n_layers):
        params.update({
            f"model.layers.{l}.input_layernorm.weight": torch.ones(embed_dim),
            f"model.layers.{l}.post_attention_layernorm.weight": torch.ones(embed_dim),
            f"model.layers.{l}.self_attn.q_proj.weight": torch.randn(embed_dim, embed_dim),
            f"model.layers.{l}.self_attn.k_proj.weight": torch.randn(16, embed_dim),
            f"model.layers.{l}.self_attn.v_proj.weight": torch.randn(16, embed_dim),
            f"model.layers.{l}.self_attn.o_proj.weight": torch.randn(embed_dim, embed_dim),
            f"model.layers.{l}.self_attn.q_norm.weight": torch.ones(8),
            f"model.layers.{l}.self_attn.k_norm.weight": torch.ones(8),
            f"model.layers.{l}.mlp.gate_proj.weight": torch.randn(d_ff, embed_dim),
            f"model.layers.{l}.mlp.up_proj.weight": torch.randn(d_ff, embed_dim),
            f"model.layers.{l}.mlp.down_proj.weight": torch.randn(embed_dim, d_ff),
        })
    return params


class TestQwen3WeightTying(unittest.TestCase):
    """Qwen3 tie_word_embeddings：官方 0.6B/1.7B/4B 为 True，8B/14B/32B/30B-A3B 为 False。"""

    def test_tied_weights_share_parameter(self):
        model = Qwen3Model(_qwen3_cfg(tie=True))
        self.assertIs(model.out_head.weight, model.tok_embed.weight)
        untied = Qwen3Model(_qwen3_cfg(tie=False))
        self.assertIsNot(untied.out_head.weight, untied.tok_embed.weight)

    def test_param_count_reduced(self):
        cfg = _qwen3_cfg(tie=True)
        tied = Qwen3Model(cfg)
        untied = Qwen3Model(_qwen3_cfg(tie=False))
        diff = (
            sum(p.numel() for p in untied.parameters())
            - sum(p.numel() for p in tied.parameters())
        )
        self.assertEqual(diff, cfg.vocab_size * cfg.embed_dim)

    def test_state_dict_single_storage(self):
        model = Qwen3Model(_qwen3_cfg(tie=True))
        state = model.state_dict()
        # 两个键都存在，但指向同一份物理存储（torch.save 时只存一份）
        self.assertIn("tok_embed.weight", state)
        self.assertIn("out_head.weight", state)
        self.assertEqual(
            state["tok_embed.weight"].data_ptr(),
            state["out_head.weight"].data_ptr(),
        )

    def test_forward_matches_manual_copy(self):
        """tie 版 forward 与"手动复制权重"的未 tie 版 allclose。"""
        torch.manual_seed(123)
        tied = Qwen3Model(_qwen3_cfg(tie=True)).eval()
        untied = Qwen3Model(_qwen3_cfg(tie=False)).eval()
        untied.load_state_dict(tied.state_dict())
        x = torch.randint(0, 64, (2, 8))
        with torch.no_grad():
            self.assertTrue(torch.allclose(tied(x), untied(x), atol=1e-6))

    def test_official_configs_tie_flags(self):
        from config.qwen3.model_cfgs import get_cfgs

        expected = {
            "0.6B": True, "1.7B": True, "4B": True,
            "8B": False, "14B": False, "32B": False, "30B-A3B": False,
        }
        for key, tied in expected.items():
            self.assertEqual(get_cfgs(key).tie_word_embeddings, tied, key)

    def test_load_weights_tie_skips_lm_head(self):
        """tie 时 load_weights_into_qwen 跳过 lm_head：重新绑定共享 Parameter。"""
        from models.io.qwen3.load_weights import load_weights_into_qwen

        cfg = ModelConfig.from_dict({
            "vocab_size": 64, "context_length": 16, "embed_dim": 32,
            "d_ff": 64, "n_heads": 4, "n_layers": 2, "n_kv_groups": 2,
            "head_dim": 8, "qk_norm": True, "rope_base": 10_000.0,
            "num_experts": 0, "dtype": torch.float32,
            "tie_word_embeddings": True,
        })
        model = Qwen3Model(cfg)
        params = _qwen3_mock_params()
        params["lm_head.weight"] = torch.randn(64, 32)   # tie 时必须被忽略
        load_weights_into_qwen(model, cfg, params)
        self.assertIs(model.out_head.weight, model.tok_embed.weight)
        self.assertTrue(
            torch.allclose(model.out_head.weight, params["model.embed_tokens.weight"])
        )

    def test_load_weights_untied_copies_embedding_when_no_lm_head(self):
        """未 tie 且 checkpoint 无 lm_head：保留"复制 embedding"兼容路径（独立 Parameter）。"""
        from models.io.qwen3.load_weights import load_weights_into_qwen

        cfg = ModelConfig.from_dict({
            "vocab_size": 64, "context_length": 16, "embed_dim": 32,
            "d_ff": 64, "n_heads": 4, "n_layers": 2, "n_kv_groups": 2,
            "head_dim": 8, "qk_norm": True, "rope_base": 10_000.0,
            "num_experts": 0, "dtype": torch.float32,
            "tie_word_embeddings": False,
        })
        model = Qwen3Model(cfg)
        params = _qwen3_mock_params()
        load_weights_into_qwen(model, cfg, params)
        self.assertIsNot(model.out_head.weight, model.tok_embed.weight)
        self.assertTrue(
            torch.allclose(model.out_head.weight, model.tok_embed.weight)
        )


if __name__ == "__main__":
    unittest.main()
