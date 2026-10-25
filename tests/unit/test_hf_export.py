"""HF 导出测试（models/io/export，reasoning ch08）。"""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

try:
    from safetensors.torch import load_file
except ImportError:
    load_file = None

try:
    import transformers  # noqa: F401
except ImportError:
    transformers = None

from models.io.export import hf_export
from models.qwen3.qwen3 import Model


def _qwen_config():
    return SimpleNamespace(
        vocab_size=64, context_length=32, embed_dim=16, d_ff=32,
        n_heads=4, n_layers=2, n_kv_groups=2, head_dim=4,
        dropout=0.0, qk_norm=True, rope_base=10000.0,
        num_experts=0, dtype=torch.float32,
    )


def _tiny_qwen3(seed=0):
    torch.manual_seed(seed)
    return Model(_qwen_config())


@unittest.skipUnless(load_file is not None, "safetensors 未安装")
class TestExportToHF(unittest.TestCase):
    def setUp(self):
        self.model = _tiny_qwen3(seed=0)
        self.cfg = _qwen_config()
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.out_dir = hf_export.export_to_hf(
            self.model, Path(tmp.name) / "hf_out", self.cfg,
        )

    def test_state_dict_keys_and_values_match(self):
        exported = load_file(str(self.out_dir / "model.safetensors"))
        expected = hf_export.local_to_hf_state_dict(self.model.state_dict())
        self.assertEqual(set(exported), set(expected))
        for key, value in expected.items():
            self.assertTrue(torch.allclose(exported[key], value))

    def test_expected_hf_qwen3_keys_present(self):
        exported = load_file(str(self.out_dir / "model.safetensors"))
        for key in (
            "model.embed_tokens.weight",
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.k_norm.weight",
            "model.layers.0.mlp.gate_proj.weight",
            "model.layers.1.input_layernorm.weight",
            "model.layers.1.post_attention_layernorm.weight",
            "model.norm.weight",
            "lm_head.weight",
        ):
            self.assertIn(key, exported)
        self.assertEqual(len(exported), len(self.model.state_dict()))

    def test_config_json_fields(self):
        config = json.loads((self.out_dir / "config.json").read_text(encoding="utf-8"))
        self.assertEqual(config["architectures"], ["Qwen3ForCausalLM"])
        self.assertEqual(config["model_type"], "qwen3")
        self.assertEqual(config["vocab_size"], 64)
        self.assertEqual(config["hidden_size"], 16)
        self.assertEqual(config["num_hidden_layers"], 2)
        self.assertEqual(config["num_attention_heads"], 4)
        self.assertEqual(config["num_key_value_heads"], 2)
        self.assertEqual(config["head_dim"], 4)
        self.assertEqual(config["intermediate_size"], 32)
        self.assertEqual(config["max_position_embeddings"], 32)
        self.assertEqual(config["rope_theta"], 10000.0)
        self.assertEqual(config["torch_dtype"], "float32")
        self.assertFalse(config["tie_word_embeddings"])
        self.assertTrue((self.out_dir / "generation_config.json").exists())

    def test_moe_config_rejected(self):
        cfg = _qwen_config()
        cfg.num_experts = 8
        with self.assertRaises(NotImplementedError):
            hf_export.export_to_hf(
                self.model, self.out_dir.parent / "moe_out", cfg,
            )


class TestLocalToHfStateDict(unittest.TestCase):
    def test_unknown_key_raises(self):
        with self.assertRaises(KeyError):
            hf_export.local_to_hf_state_dict({"weird.key": torch.zeros(1)})


@unittest.skipUnless(transformers is not None, "transformers 未安装")
class TestTinyLLMWrapper(unittest.TestCase):
    def test_wrapper_forward_allclose(self):
        local = _tiny_qwen3(seed=0)
        wrapper = hf_export.TinyLLMForCausalLM(
            hf_export.TinyLLMConfig(model_cfg=vars(_qwen_config()))
        )
        wrapper.model.load_state_dict(local.state_dict())
        wrapper.eval()
        local.eval()
        input_ids = torch.randint(0, 64, (2, 7))
        with torch.no_grad():
            output = wrapper(input_ids)
            reference = local(input_ids)
        self.assertTrue(torch.allclose(output.logits, reference, atol=1e-6))

    def test_wrapper_labels_loss_finite(self):
        wrapper = hf_export.TinyLLMForCausalLM(
            hf_export.TinyLLMConfig(model_cfg=vars(_qwen_config()))
        )
        input_ids = torch.randint(0, 64, (2, 7))
        labels = torch.randint(0, 64, (2, 7))
        output = wrapper(input_ids, labels=labels)
        self.assertTrue(torch.isfinite(output.loss))


if __name__ == "__main__":
    unittest.main()
