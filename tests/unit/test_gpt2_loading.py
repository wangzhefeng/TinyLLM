import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from models.gpt2.gpt2_124M import Model
from models.io.gpt2.openai_gpt2_weights_load import load_weights_download
from models.io.gpt2.openai_gpt2_weights_load_hf import load_weights_hf
from models.io.gpt2.openai_gpt2_weights_load_hf_safetensors import (
    load_weights_hf_safetensors,
)

VOCAB, CTX, EMB, D_FF, N_LAYERS = 128, 16, 32, 64, 2


def _tiny_cfg():
    return SimpleNamespace(
        vocab_size=VOCAB, context_length=CTX, embed_dim=EMB, d_ff=D_FF,
        n_heads=4, n_layers=N_LAYERS, dropout=0.0, qkv_bias=True,
        dtype=torch.float32, is_train=False,
    )


def _hf_state_dict():
    g = torch.Generator().manual_seed(123)
    rnd = lambda *s: torch.randn(*s, generator=g)
    d = {
        "wte.weight": rnd(VOCAB, EMB),
        "wpe.weight": rnd(CTX, EMB),
        "ln_f.weight": rnd(EMB), "ln_f.bias": rnd(EMB),
    }
    for b in range(N_LAYERS):
        for k, v in {
            f"h.{b}.attn.c_attn.weight": rnd(EMB, 3 * EMB),
            f"h.{b}.attn.c_attn.bias": rnd(3 * EMB),
            f"h.{b}.attn.c_proj.weight": rnd(EMB, EMB),
            f"h.{b}.attn.c_proj.bias": rnd(EMB),
            f"h.{b}.mlp.c_fc.weight": rnd(EMB, D_FF),
            f"h.{b}.mlp.c_fc.bias": rnd(D_FF),
            f"h.{b}.mlp.c_proj.weight": rnd(D_FF, EMB),
            f"h.{b}.mlp.c_proj.bias": rnd(EMB),
            f"h.{b}.ln_1.weight": rnd(EMB), f"h.{b}.ln_1.bias": rnd(EMB),
            f"h.{b}.ln_2.weight": rnd(EMB), f"h.{b}.ln_2.bias": rnd(EMB),
        }.items():
            d[k] = v
    return d


def _assert_core_mappings(test, model, d):
    torch.testing.assert_close(model.tok_embed.weight, d["wte.weight"])
    torch.testing.assert_close(model.pos_embed.weight, d["wpe.weight"])
    torch.testing.assert_close(model.out_head.weight, d["wte.weight"])
    torch.testing.assert_close(model.final_norm.scale, d["ln_f.weight"])
    q_w = d["h.0.attn.c_attn.weight"].split(EMB, dim=-1)[0]
    torch.testing.assert_close(model.trf_blocks[0].attn.W_query.weight, q_w.T)
    torch.testing.assert_close(model.trf_blocks[0].norm1.scale, d["h.0.ln_1.weight"])


class TestLoadWeightsHF(unittest.TestCase):
    def test_hf_state_dict_lands_on_current_module_names(self):
        model = Model(_tiny_cfg())
        d = _hf_state_dict()
        gpt_hf = SimpleNamespace(state_dict=lambda: d)
        load_weights_hf(model, gpt_hf, {"n_layers": N_LAYERS})
        _assert_core_mappings(self, model, d)
        logits = model(torch.randint(0, VOCAB, (1, 8)))
        self.assertEqual(logits.shape, (1, 8, VOCAB))


class TestLoadWeightsSafetensors(unittest.TestCase):
    def test_safetensors_params_lands_on_current_module_names(self):
        model = Model(_tiny_cfg())
        d = _hf_state_dict()
        load_weights_hf_safetensors(model, d)
        _assert_core_mappings(self, model, d)


class TestLoadWeightsTF(unittest.TestCase):
    def test_openai_tf_params_lands_on_current_module_names(self):
        model = Model(_tiny_cfg())
        d = _hf_state_dict()
        rng = np.random.default_rng(123)
        params = {
            "wte": rng.standard_normal((VOCAB, EMB)).astype(np.float32),
            "wpe": rng.standard_normal((CTX, EMB)).astype(np.float32),
            "g": rng.standard_normal(EMB).astype(np.float32),
            "b": rng.standard_normal(EMB).astype(np.float32),
            "blocks": [
                {
                    "attn": {
                        "c_attn": {"w": rng.standard_normal((EMB, 3 * EMB)).astype(np.float32),
                                   "b": rng.standard_normal(3 * EMB).astype(np.float32)},
                        "c_proj": {"w": rng.standard_normal((EMB, EMB)).astype(np.float32),
                                   "b": rng.standard_normal(EMB).astype(np.float32)},
                    },
                    "mlp": {
                        "c_fc": {"w": rng.standard_normal((EMB, D_FF)).astype(np.float32),
                                 "b": rng.standard_normal(D_FF).astype(np.float32)},
                        "c_proj": {"w": rng.standard_normal((D_FF, EMB)).astype(np.float32),
                                   "b": rng.standard_normal(EMB).astype(np.float32)},
                    },
                    "ln_1": {"g": rng.standard_normal(EMB).astype(np.float32),
                             "b": rng.standard_normal(EMB).astype(np.float32)},
                    "ln_2": {"g": rng.standard_normal(EMB).astype(np.float32),
                             "b": rng.standard_normal(EMB).astype(np.float32)},
                }
                for _ in range(N_LAYERS)
            ],
        }
        load_weights_download(model, params)
        np.testing.assert_allclose(
            model.tok_embed.weight.detach().numpy(), params["wte"], rtol=1e-6,
        )
        np.testing.assert_allclose(
            model.pos_embed.weight.detach().numpy(), params["wpe"], rtol=1e-6,
        )
        np.testing.assert_allclose(
            model.out_head.weight.detach().numpy(), params["wte"], rtol=1e-6,
        )
        np.testing.assert_allclose(
            model.final_norm.scale.detach().numpy(), params["g"], rtol=1e-6,
        )


class TestLoadModelSmoke(unittest.TestCase):
    @unittest.skipUnless(
        Path("downloaded_models/gpt2_model/models--openai-community--gpt2").exists(),
        "需要本地 gpt2 HF 缓存",
    )
    def test_load_model_gpt2_124m_generates(self):
        from inference.generation import generate_text
        from inference.loading import load_model

        model, config, extras = load_model("gpt2-124m", device="cpu")
        text = generate_text(
            model, "Every effort moves", family="gpt2",
            max_new_tokens=8, temperature=1.0, top_k=1, eos_token_id=50256,
            device="cpu",
        )
        self.assertTrue(len(text) > 0)


if __name__ == "__main__":
    unittest.main()
