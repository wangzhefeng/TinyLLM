import unittest
from types import SimpleNamespace

import torch


class TestModelRegistry(unittest.TestCase):
    def test_list_models_and_spec(self):
        from inference import MODEL_REGISTRY, get_model_spec, list_models
        keys = list_models()
        self.assertIn("gpt2-124m", keys)
        self.assertIn("qwen3-0.6b", keys)
        spec = get_model_spec("gpt2-124m")
        self.assertEqual(spec.family, "gpt2")
        self.assertEqual(spec.eos_token_id, 50256)
        # gemma3 标记为 WIP
        self.assertFalse(MODEL_REGISTRY["gemma3-270m"].available)

    def test_unknown_key_error(self):
        from inference import get_model_spec
        with self.assertRaises(KeyError) as cm:
            get_model_spec("gpt5")
        self.assertIn("gpt2-124m", str(cm.exception))

    def test_public_api_lazy_no_torch_at_import(self):
        # import inference 不应拉起 torch（懒加载契约）
        import subprocess
        import sys
        code = (
            "import sys; sys.path.insert(0, '.');"
            "import inference;"
            "assert 'torch' not in sys.modules, 'torch should be lazy';"
            "assert inference.list_models();"
            "print('LAZY_OK')"
        )
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("LAZY_OK", result.stdout)


class TestLoadModelErrors(unittest.TestCase):
    def test_unavailable_model(self):
        from inference import load_model
        with self.assertRaises(NotImplementedError):
            load_model("gemma3-270m")

    def test_bad_source(self):
        from inference import load_model
        with self.assertRaises(ValueError):
            load_model("gpt2-124m", source="gguf")


class TestGenerateTextFacade(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)
        from models.gpt2.gpt2_124M import Model
        self.cfg = SimpleNamespace(
            vocab_size=50257, context_length=32, embed_dim=32, d_ff=64,
            n_heads=4, n_layers=2, dropout=0.0, qkv_bias=True,
            dtype=torch.float32, is_train=False,
        )
        self.model = Model(self.cfg).eval()

    def test_gpt2_facade_offline(self):
        from inference import generate_text
        text = generate_text(
            self.model, "Hello, I am", family="gpt2",
            max_new_tokens=4, context_length=32,
        )
        self.assertIsInstance(text, str)
        self.assertFalse(text.startswith("Hello, I am"))
        self.assertTrue(text)

    def test_gpt2_facade_supports_cache(self):
        # 2026-10-20 起 GPT-2 的 MultiHeadAttention 支持 KV cache，facade 不再拒绝
        from inference import generate_text
        text = generate_text(self.model, "hi", family="gpt2", max_new_tokens=2,
                             context_length=32, use_cache=True)
        self.assertIsInstance(text, str)
        self.assertFalse(text.startswith("hi"))
        self.assertTrue(text)

    def test_llama_facade_rejects_cache(self):
        # llama 族仍是 nn.Sequential 堆叠，不支持 KV cache
        from inference import generate_text
        with self.assertRaises(ValueError):
            generate_text(self.model, "hi", family="llama", max_new_tokens=2,
                          context_length=32, use_cache=True)


if __name__ == "__main__":
    unittest.main()
