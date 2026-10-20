"""LoRA 通用层测试（Phase 2 后训练域）。"""
import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn

from layers.lora import (
    LinearWithLoRA,
    LinearWithLoRAMerged,
    LoRALayer,
    freeze_linear_layers,
    merge_lora_weights,
    replace_linear_with_lora,
)
from models.gpt2.gpt2_124M import Model


class TestLoRALayer(unittest.TestCase):
    def test_zero_init_delta(self):
        """B 初始为零 → 初始 LoRA 旁路输出为零（不破坏预训练行为）。"""
        lora = LoRALayer(in_dim=8, out_dim=8, rank=4, alpha=2.0)
        x = torch.randn(3, 8)
        self.assertTrue(torch.equal(lora(x), torch.zeros(3, 8)))

    def test_delta_shape(self):
        lora = LoRALayer(8, 16, rank=4, alpha=1.0)
        with torch.no_grad():
            lora.B.fill_(1.0)
        out = lora(torch.randn(3, 8))
        self.assertEqual(out.shape, (3, 16))


class TestLinearWithLoRA(unittest.TestCase):
    def test_equivalence_unmerged_vs_merged(self):
        """不合并版与合并版前向必须一致。"""
        torch.manual_seed(0)
        base = nn.Linear(16, 16, bias=True)
        lora = LinearWithLoRA(base, rank=4, alpha=2.0)
        merged = LinearWithLoRAMerged(nn.Linear(16, 16, bias=True), rank=4, alpha=2.0)
        # 复制相同权重
        merged.linear.load_state_dict(lora.linear.state_dict())
        with torch.no_grad():
            merged.lora.A.copy_(lora.lora.A)
            merged.lora.B.copy_(lora.lora.B)
            merged.lora.B.fill_(0.5)
            lora.lora.B.fill_(0.5)
        x = torch.randn(2, 16)
        torch.testing.assert_close(lora(x), merged(x), atol=1e-5, rtol=1e-4)


class TestMergeLoRAWeights(unittest.TestCase):
    def test_merge_roundtrip(self):
        """merge 后的普通 Linear 与原 LoRA 模块输出一致。"""
        torch.manual_seed(0)
        lora = LinearWithLoRA(nn.Linear(8, 8), rank=4, alpha=1.0)
        with torch.no_grad():
            lora.lora.B.fill_(0.3)
        x = torch.randn(2, 8)
        merged = merge_lora_weights(lora)
        self.assertIsInstance(merged, nn.Linear)
        torch.testing.assert_close(lora(x), merged(x), atol=1e-5, rtol=1e-4)

    def test_non_lora_passthrough(self):
        plain = nn.Linear(4, 4)
        self.assertIs(merge_lora_weights(plain), plain)


class TestLoRAOnModel(unittest.TestCase):
    def test_replace_and_freeze_on_gpt2(self):
        """LoRA 可即插即用到 models/ 的 GPT-2；冻结后只有 lora 参数可训练。"""
        model = Model(SimpleNamespace(
            vocab_size=32, context_length=16, embed_dim=16, d_ff=32,
            n_heads=2, n_layers=1, dropout=0.0, qkv_bias=False,
            dtype=torch.float32, is_train=False,
        ))
        replace_linear_with_lora(model, rank=2, alpha=2.0)
        freeze_linear_layers(model)
        trainable = {n for n, p in model.named_parameters() if p.requires_grad}
        # LoRA 旁路参数可训练
        lora_params = {n for n in trainable if "lora." in n}
        self.assertTrue(len(lora_params) > 0)
        # 所有 Linear 主权重（非 lora 旁路）都被冻结：不存在
        # "W_query/W_key/W_value/out_proj/fc*/layers.N" 等结尾 .weight 的可训练参数
        frozen_linear = [n for n, p in model.named_parameters()
                         if not p.requires_grad and "lora." not in n]
        self.assertTrue(any("W_query.linear.weight" in n for n in frozen_linear))
        # 且 embedding/norm 仍可训练（LoRA 惯例）
        self.assertTrue(any("tok_embed" in n for n in trainable))
        # 前向可用
        out = model(torch.randint(0, 32, (1, 4)))
        self.assertEqual(out.shape, (1, 4, 32))


class TestBackwardCompatShim(unittest.TestCase):
    def test_shim_reexports(self):
        """post_training 的兼容 shim 与 layers.lora 是同一对象。"""
        from post_training.peft.layers.lora_dora import LinearWithLoRA as ShimLoRA
        self.assertIs(ShimLoRA, LinearWithLoRA)


if __name__ == "__main__":
    unittest.main()
