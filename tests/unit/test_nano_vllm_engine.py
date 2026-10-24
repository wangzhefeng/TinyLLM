# -*- coding: utf-8 -*-

# ***************************************************
# * File        : test_nano_vllm_engine.py
# * Description : nano_vllm 引擎层回归测试：
# *               1. Config.from_pretrained 正确读取 HF config.json /
# *                  generation_config.json（覆盖 os import 回归）；
# *               2. LLMEngine 传入外部 model 且 tensor_parallel_size > 1 时
# *                  显式 raise（worker 子进程无法共享主进程权重）。
# * Requirement : torch >= 2.0
# ***************************************************

import json
import tempfile
import unittest
from pathlib import Path

from inference.serving.nano_vllm.config import Config
from inference.serving.nano_vllm.engine.llm_engine import LLMEngine


def _write_hf_model_dir(tmp: str, with_generation_config: bool = True) -> str:
    d = Path(tmp)
    (d / "config.json").write_text(json.dumps({
        "vocab_size": 128,
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "max_position_embeddings": 256,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "attention_bias": False,
        "tie_word_embeddings": False,
        "hidden_act": "silu",
        "eos_token_id": 7,
    }), encoding="utf-8")
    if with_generation_config:
        (d / "generation_config.json").write_text(
            json.dumps({"eos_token_id": [7, 8]}), encoding="utf-8",
        )
    return str(d)


class TestConfigFromPretrained(unittest.TestCase):

    def test_reads_hf_config_and_generation_eos(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = Config.from_pretrained(_write_hf_model_dir(tmp))
        self.assertEqual(config.vocab_size, 128)
        self.assertEqual(config.hidden_size, 64)
        self.assertEqual(config.num_key_value_heads, 2)
        # generation_config.json 优先；list 形式取最后一个（Qwen3 惯例，末位为真正的结束符）
        self.assertEqual(config.eos, 8)

    def test_falls_back_to_config_eos_and_applies_overrides(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = Config.from_pretrained(
                _write_hf_model_dir(tmp, with_generation_config=False),
                max_num_seqs=8,
            )
        self.assertEqual(config.eos, 7)
        self.assertEqual(config.max_num_seqs, 8)


class TestLLMEngineExternalModelGuard(unittest.TestCase):

    def test_external_model_with_tp_greater_than_1_raises(self):
        config = Config(
            vocab_size=64, hidden_size=128, intermediate_size=256,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            head_dim=32, max_position_embeddings=64, max_model_len=32,
            max_num_seqs=2, eos=0, tensor_parallel_size=2,
        )
        with self.assertRaisesRegex(ValueError, "外部 model"):
            LLMEngine(config, model=object())


if __name__ == "__main__":
    unittest.main()
