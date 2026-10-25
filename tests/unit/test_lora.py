"""LoRA 通用层测试（Phase 2 后训练域）。"""
import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn

from layers.lora import (
    LinearWithDoRA,
    LinearWithDoRACorrect,
    LinearWithDoRAMerged,
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


class TestAlphaRankScaling(unittest.TestCase):
    """缩放语义：实际缩放为 alpha/rank（LoRA 论文 / HF peft / 附录 E 最新版）。"""

    def test_scaling_is_alpha_over_rank(self):
        """rank=4, alpha=8 → 有效缩放 = 2，而非裸 alpha = 8。"""
        torch.manual_seed(0)
        lora = LoRALayer(8, 8, rank=4, alpha=8.0)
        with torch.no_grad():
            lora.A.normal_()
            lora.B.fill_(1.0)
        x = torch.randn(3, 8)
        manual = 2.0 * (x @ lora.A @ lora.B)
        torch.testing.assert_close(lora(x), manual)

    def test_dropout_zero_keeps_plain_behavior(self):
        """dropout=0.0（默认）不引入任何行为变化。"""
        torch.manual_seed(1)
        lora = LoRALayer(6, 5, rank=2, alpha=4.0, dropout=0.0)
        lora.eval()
        x = torch.randn(2, 6)
        manual = (4.0 / 2) * (x @ lora.A @ lora.B)
        torch.testing.assert_close(lora(x), manual)

    def test_dropout_only_active_in_train_mode(self):
        """dropout>0 在 eval 模式为恒等；train 模式改变旁路输入。"""
        torch.manual_seed(2)
        lora = LoRALayer(6, 5, rank=2, alpha=4.0, dropout=0.5)
        x = torch.randn(2, 6)
        lora.eval()
        manual = (4.0 / 2) * (x @ lora.A @ lora.B)
        torch.testing.assert_close(lora(x), manual)


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

    def test_merge_lora_merged_variant(self):
        """LinearWithLoRAMerged 也可折叠为普通 Linear（输出一致）。"""
        torch.manual_seed(0)
        lora = LinearWithLoRAMerged(nn.Linear(8, 8), rank=4, alpha=2.0)
        with torch.no_grad():
            lora.lora.B.fill_(0.3)
        x = torch.randn(2, 8)
        merged = merge_lora_weights(lora)
        self.assertIsInstance(merged, nn.Linear)
        torch.testing.assert_close(lora(x), merged(x), atol=1e-5, rtol=1e-4)


def _build_tiny_gpt():
    """与 models/gpt2 兼容的最小 GPT（属性名与真实模型一致）。"""
    return Model(SimpleNamespace(
        vocab_size=32, context_length=16, embed_dim=16, d_ff=32,
        n_heads=2, n_layers=1, dropout=0.0, qkv_bias=False,
        dtype=torch.float32, is_train=False,
    ))


class TestTargetSelection(unittest.TestCase):
    """replace_linear_with_lora 的目标层过滤。"""

    def test_default_replaces_all(self):
        model = _build_tiny_gpt()
        replace_linear_with_lora(model, rank=2, alpha=2.0)
        self.assertIsInstance(model.out_head, LinearWithLoRA)
        self.assertIsInstance(model.trf_blocks[0].attn.W_query, LinearWithLoRA)

    def test_exclude_keywords_spares_out_head(self):
        """exclude_keywords=["out_head"]：分类头保持普通 Linear（clf lora 用法）。"""
        model = _build_tiny_gpt()
        replace_linear_with_lora(model, rank=2, alpha=2.0,
                                 exclude_keywords=["out_head"])
        self.assertIsInstance(model.out_head, nn.Linear)
        self.assertNotIsInstance(model.out_head, LinearWithLoRA)
        self.assertIsInstance(model.trf_blocks[0].attn.W_query, LinearWithLoRA)

    def test_target_keywords_only_attention_qkv(self):
        """target_keywords：只给注意力 QKV 投影加 LoRA。"""
        model = _build_tiny_gpt()
        replace_linear_with_lora(model, rank=2, alpha=2.0,
                                 target_keywords=["W_query", "W_key", "W_value"])
        self.assertIsInstance(model.trf_blocks[0].attn.W_query, LinearWithLoRA)
        self.assertIsInstance(model.trf_blocks[0].attn.W_key, LinearWithLoRA)
        self.assertIsInstance(model.trf_blocks[0].attn.W_value, LinearWithLoRA)
        self.assertNotIsInstance(model.trf_blocks[0].attn.out_proj, LinearWithLoRA)
        self.assertNotIsInstance(model.trf_blocks[0].ff.layers[0], LinearWithLoRA)
        self.assertNotIsInstance(model.out_head, LinearWithLoRA)


class TestLoRAOnModel(unittest.TestCase):
    def test_replace_and_freeze_on_gpt2(self):
        """LoRA 可即插即用到 models/ 的 GPT-2；冻结后只有 lora 参数可训练。"""
        model = _build_tiny_gpt()
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


class TestDoRA(unittest.TestCase):
    """DoRA 变体：简化版不等价回归 + 论文版 LinearWithDoRACorrect。"""

    def _same_weight_dora_pair(self):
        torch.manual_seed(0)
        base1 = nn.Linear(8, 8, bias=True)
        base2 = nn.Linear(8, 8, bias=True)
        base2.load_state_dict(base1.state_dict())
        dora = LinearWithDoRA(base1, rank=4, alpha=2.0)
        dora_merged = LinearWithDoRAMerged(base2, rank=4, alpha=2.0)
        with torch.no_grad():
            dora_merged.lora.A.copy_(dora.lora.A)
            dora.lora.B.fill_(0.5)
            dora_merged.lora.B.fill_(0.5)
        return dora, dora_merged

    def test_simplified_variants_inequivalent(self):
        """已知简化（回归标注）：LinearWithDoRA 与 LinearWithDoRAMerged 数学不等价。

        前者只归一化 LoRA 旁路激活（加在未归一化的 Wx 上），后者对组合权重
        W+ΔW 做 m·V 分解——两者来自书附录 E 简化版的两个不同侧面，
        merge 前后行为不同。若本测试失败说明二者被统一（例如都改为论文版），
        请同步更新 layers/lora.py 的 docstring 与 docs/20-post-training/peft.md。
        """
        dora, dora_merged = self._same_weight_dora_pair()
        x = torch.randn(2, 8)
        with self.assertRaises(AssertionError):
            torch.testing.assert_close(dora(x), dora_merged(x))

    def test_merge_dora_raises(self):
        """merge_lora_weights 对 DoRA 变体显式 TypeError（合并是 m·V 重建）。"""
        dora = LinearWithDoRA(nn.Linear(8, 8), rank=4, alpha=2.0)
        with self.assertRaises(TypeError):
            merge_lora_weights(dora)
        dora_merged = LinearWithDoRAMerged(nn.Linear(8, 8), rank=4, alpha=2.0)
        with self.assertRaises(TypeError):
            merge_lora_weights(dora_merged)

    def test_dora_correct_init_matches_base(self):
        """论文版：B=0 → ΔW=0 → W' = m·V = W，初始不改变预训练行为。"""
        torch.manual_seed(0)
        base = nn.Linear(8, 6, bias=True)
        dora_c = LinearWithDoRACorrect(base, rank=4, alpha=2.0)
        x = torch.randn(3, 8)
        torch.testing.assert_close(dora_c(x), base(x), rtol=1e-4, atol=1e-4)

    def test_dora_correct_magnitude_per_output_neuron(self):
        """论文版：m ∈ R^{out}（每个输出神经元一个），范数沿 in 维（dim=1）。"""
        base = nn.Linear(8, 6)
        dora_c = LinearWithDoRACorrect(base, rank=4, alpha=2.0)
        self.assertEqual(dora_c.m.shape, (6,))
        self.assertTrue(torch.allclose(dora_c.m, base.weight.norm(p=2, dim=1)))

    def test_dora_correct_forward_trainable(self):
        """论文版前向可反传，梯度只经由 m / A / B /（未冻结的）基座流动。"""
        torch.manual_seed(0)
        base = nn.Linear(8, 6, bias=False)
        for p in base.parameters():
            p.requires_grad = False
        dora_c = LinearWithDoRACorrect(base, rank=4, alpha=2.0)
        with torch.no_grad():
            dora_c.lora.B.fill_(0.3)
        x = torch.randn(3, 8)
        dora_c(x).sum().backward()
        self.assertIsNotNone(dora_c.m.grad)
        self.assertIsNotNone(dora_c.lora.A.grad)
        self.assertIsNotNone(dora_c.lora.B.grad)


class TestBackwardCompatShim(unittest.TestCase):
    def test_merge_adapter_weights_reexports(self):
        """merge_adapter_weights（修正拼写后的转发入口）转发到 layers.lora。"""
        from post_training.peft.layers.merge_adapter_weights import (
            merge_lora_weights as shim_merge,
        )
        self.assertIs(shim_merge, merge_lora_weights)


class TestClfHeadFinetuneMethods(unittest.TestCase):
    """clf_head.finetune_model 四种策略的冻结语义。"""

    def _build(self):
        from post_training.sft.layers.clf_head import finetune_model
        return finetune_model

    def test_last_layer_only_head_trainable(self):
        from post_training.sft.layers.clf_head import finetune_model
        model = _build_tiny_gpt()
        finetune_model(model, emb_dim=16, num_classes=2, finetune_method="last_layer")
        trainable = {n for n, p in model.named_parameters() if p.requires_grad}
        self.assertTrue(trainable)
        self.assertTrue(all(n.startswith("out_head.") for n in trainable))

    def test_last_transformer_stage(self):
        from post_training.sft.layers.clf_head import finetune_model
        model = _build_tiny_gpt()
        finetune_model(model, emb_dim=16, num_classes=2,
                       finetune_method="last_transformer_stage")
        trainable = {n for n, p in model.named_parameters() if p.requires_grad}
        self.assertTrue(any(n.startswith("out_head.") for n in trainable))
        self.assertTrue(any(n.startswith("trf_blocks.0.") for n in trainable))
        self.assertTrue(any(n.startswith("final_norm.") for n in trainable))
        # 前面的层仍冻结
        frozen = {n for n, p in model.named_parameters() if not p.requires_grad}
        self.assertTrue(any("tok_embed" in n for n in frozen))

    def test_all_layers(self):
        from post_training.sft.layers.clf_head import finetune_model
        model = _build_tiny_gpt()
        finetune_model(model, emb_dim=16, num_classes=2, finetune_method="all_layers")
        for n, p in model.named_parameters():
            self.assertTrue(p.requires_grad, f"{n} 应可训练")

    def test_lora_keeps_head_plain_and_base_frozen(self):
        """lora：out_head 不被 LoRA 包裹且全量可训；基座 Linear 冻结、只训旁路。"""
        from layers.lora import LinearWithLoRA
        from post_training.sft.layers.clf_head import finetune_model
        model = _build_tiny_gpt()
        finetune_model(model, emb_dim=16, num_classes=2, finetune_method="lora")
        # 分类头保持普通 Linear（不被包裹）
        self.assertIsInstance(model.out_head, nn.Linear)
        self.assertNotIsInstance(model.out_head, LinearWithLoRA)
        trainable = {n for n, p in model.named_parameters() if p.requires_grad}
        # 可训练 = out_head 全量 + LoRA A/B；最后 block 不解冻
        self.assertTrue(any(n.startswith("out_head.") for n in trainable))
        self.assertTrue(any(".lora.A" in n or ".lora.B" in n for n in trainable))
        base_trainable = [n for n in trainable
                          if not n.startswith("out_head.") and "lora." not in n]
        self.assertEqual(base_trainable, [])
        # 基座主权重冻结（含最后一个 block）
        frozen = {n for n, p in model.named_parameters() if not p.requires_grad}
        self.assertTrue(any("trf_blocks.0.attn.W_query.linear.weight" in n for n in frozen))

    def test_unknown_method_raises(self):
        from post_training.sft.layers.clf_head import finetune_model
        model = _build_tiny_gpt()
        with self.assertRaises(ValueError):
            finetune_model(model, emb_dim=16, num_classes=2, finetune_method="typo")


if __name__ == "__main__":
    unittest.main()
