"""指令 SFT 底座路由测试：默认 GPT-2，可选 phi-1.5（ch07 路线）。"""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from models.gpt2.gpt2_124M import Model as GPT2Model
from post_training.sft.exp.exp_finetune_gpt_instruction_flow import (
    ModelFinetuningInstructionFlow,
)


def _tiny_model():
    torch.manual_seed(0)
    cfg = SimpleNamespace(
        vocab_size=32, context_length=8, embed_dim=16, d_ff=32,
        n_heads=2, n_layers=1, dropout=0.0, qkv_bias=True, dtype=torch.float32,
    )
    return GPT2Model(cfg)


def _make_exp(backbone):
    # 绕过 __init__（避免数据/权重下载），只测底座路由
    exp = object.__new__(ModelFinetuningInstructionFlow)
    exp.args = SimpleNamespace(
        backbone=backbone,
        pretrained_model_source="hf",
        pretrained_model_path=None,
    )
    exp.device = torch.device("cpu")
    exp._load_model = lambda m: m
    return exp


class TestBackboneRouting(unittest.TestCase):
    def test_phi15_routing(self):
        """backbone=phi15 时走 phi-1.5 加载器并传入本地权重目录。"""
        exp = _make_exp("phi15")
        fake = _tiny_model()
        with patch("models.io.phi.load_phi15_weights.model_with_phi15_weights") as phi_load:
            phi_load.return_value = (fake, "fake_phi_cfg")
            model, cfg = exp._load_backbone()
        phi_load.assert_called_once_with(model_path=None)
        self.assertIs(model, fake)
        self.assertEqual(cfg, "fake_phi_cfg")

    def test_gpt2_default_routing(self):
        """缺省（backbone=gpt2）行为不变：走 model_with_gpt2_weights。"""
        exp = _make_exp("gpt2")
        fake = _tiny_model()
        with patch(
            "post_training.sft.exp.exp_finetune_gpt_instruction_flow."
            "model_with_gpt2_weights"
        ) as gpt2_load:
            gpt2_load.return_value = (fake, "fake_gpt2_cfg")
            model, cfg = exp._load_backbone()
        gpt2_load.assert_called_once()
        self.assertIs(model, fake)
        self.assertEqual(cfg, "fake_gpt2_cfg")


if __name__ == "__main__":
    unittest.main()
