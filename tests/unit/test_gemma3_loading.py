"""Gemma3 权重加载闭环测试（合成权重，无网络）。"""
import unittest
from types import SimpleNamespace

import torch

from models.gemma3.Gemma3_270M import Model
from models.io.gemma3.load_weights import load_weights_into_gemma


GEMMA3_TINY_CONFIG = {
    "vocab_size": 100,
    "context_length": 32,
    "embed_dim": 32,
    "n_heads": 2,
    "n_layers": 2,
    "hidden_dim": 64,
    "d_ff": 64,
    "head_dim": 16,
    "qk_norm": True,
    "n_kv_groups": 1,
    "rope_local_base": 10_000.0,
    "rope_base": 1_000_000.0,
    "sliding_window": 8,
    "layer_types": ["sliding_attention", "full_attention"],
    "dtype": torch.float32,
    "query_pre_attn_scalar": 16,
}


def _build_model():
    torch.manual_seed(0)
    return Model(SimpleNamespace(**GEMMA3_TINY_CONFIG))


def _synthetic_params(with_lm_head: bool):
    """按 HF 官方命名构造形状正确的合成权重。"""
    cfg = GEMMA3_TINY_CONFIG
    torch.manual_seed(1)
    params = {
        "model.embed_tokens.weight": torch.randn(cfg["vocab_size"], cfg["embed_dim"]),
    }
    for l in range(cfg["n_layers"]):
        p = f"model.layers.{l}"
        params[f"{p}.self_attn.q_proj.weight"] = torch.randn(
            cfg["n_heads"] * cfg["head_dim"], cfg["embed_dim"])
        params[f"{p}.self_attn.k_proj.weight"] = torch.randn(
            cfg["n_kv_groups"] * cfg["head_dim"], cfg["embed_dim"])
        params[f"{p}.self_attn.v_proj.weight"] = torch.randn(
            cfg["n_kv_groups"] * cfg["head_dim"], cfg["embed_dim"])
        params[f"{p}.self_attn.o_proj.weight"] = torch.randn(
            cfg["n_heads"] * cfg["head_dim"], cfg["n_heads"] * cfg["head_dim"])
        params[f"{p}.self_attn.q_norm.weight"] = torch.rand(cfg["head_dim"]) + 0.5
        params[f"{p}.self_attn.k_norm.weight"] = torch.rand(cfg["head_dim"]) + 0.5
        params[f"{p}.mlp.gate_proj.weight"] = torch.randn(cfg["d_ff"], cfg["embed_dim"])
        params[f"{p}.mlp.up_proj.weight"] = torch.randn(cfg["d_ff"], cfg["embed_dim"])
        params[f"{p}.mlp.down_proj.weight"] = torch.randn(cfg["embed_dim"], cfg["d_ff"])
        params[f"{p}.input_layernorm.weight"] = torch.rand(cfg["embed_dim"]) + 0.5
        params[f"{p}.post_attention_layernorm.weight"] = torch.rand(cfg["embed_dim"]) + 0.5
        params[f"{p}.pre_feedforward_layernorm.weight"] = torch.rand(cfg["embed_dim"]) + 0.5
        params[f"{p}.post_feedforward_layernorm.weight"] = torch.rand(cfg["embed_dim"]) + 0.5
    params["model.norm.weight"] = torch.rand(cfg["embed_dim"]) + 0.5
    if with_lm_head:
        params["lm_head.weight"] = torch.randn(cfg["vocab_size"], cfg["embed_dim"])
    return params


class TestGemma3WeightLoading(unittest.TestCase):
    def test_load_with_tying(self):
        """官方 270m 无 lm_head.weight：走 tying 分支。"""
        model = _build_model()
        params = _synthetic_params(with_lm_head=False)
        load_weights_into_gemma(model, GEMMA3_TINY_CONFIG, params)
        # tying 保持：out_head 与 tok_embed 是同一参数
        self.assertIs(model.out_head.weight, model.tok_embed.weight)
        torch.testing.assert_close(
            model.tok_embed.weight.detach(),
            params["model.embed_tokens.weight"].to(model.tok_embed.weight.dtype),
        )
        # 抽查块内权重赋值生效
        torch.testing.assert_close(
            model.blocks[1].attn.out_proj.weight.detach(),
            params["model.layers.1.self_attn.o_proj.weight"].to(model.blocks[1].attn.out_proj.weight.dtype),
        )
        torch.testing.assert_close(
            model.blocks[0].ff.fc1.weight.detach(),
            params["model.layers.0.mlp.gate_proj.weight"].to(model.blocks[0].ff.fc1.weight.dtype),
        )

    def test_load_with_lm_head(self):
        """带 lm_head.weight 的权重也能加载（最后赋值生效）。"""
        model = _build_model()
        params = _synthetic_params(with_lm_head=True)
        load_weights_into_gemma(model, GEMMA3_TINY_CONFIG, params)
        torch.testing.assert_close(
            model.out_head.weight.detach(),
            params["lm_head.weight"].to(model.out_head.weight.dtype),
        )

    def test_forward_after_load(self):
        model = _build_model().eval()
        load_weights_into_gemma(model, GEMMA3_TINY_CONFIG, _synthetic_params(False))
        x = torch.randint(0, GEMMA3_TINY_CONFIG["vocab_size"], (2, 8))
        with torch.no_grad():
            logits = model(x)
        self.assertEqual(
            logits.shape, (2, 8, GEMMA3_TINY_CONFIG["vocab_size"]))
        self.assertTrue(torch.isfinite(logits.float()).all())

    def test_registry_entry_available(self):
        from models.registry import get_model_spec

        spec = get_model_spec("gemma3-270m")
        self.assertTrue(spec.available)
        self.assertEqual(spec.family, "gemma3")
        self.assertEqual(spec.eos_token_id, 1)


if __name__ == "__main__":
    unittest.main()
