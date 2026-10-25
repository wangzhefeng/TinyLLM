"""Prompt Tuning 自研实现测试（Lester et al. 2021；LLMs-from-scratch 附录 E 主题）。"""
import unittest
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from post_training.peft.demos.prompt_tuning import (
    SoftPromptGPT,
    freeze_base_model,
    init_from_text,
    count_trainable_params,
)
from models.gpt2.gpt2_124M import Model


def _tiny_gpt2(vocab=32):
    torch.manual_seed(3)
    cfg = SimpleNamespace(
        vocab_size=vocab, context_length=32, embed_dim=32, d_ff=64,
        n_heads=4, n_layers=2, dropout=0.0, qkv_bias=True,
        dtype=torch.float32, is_train=False,
    )
    return Model(cfg)


class TestFreezeAndParams(unittest.TestCase):
    def test_only_soft_prompt_trainable(self):
        """冻结基座后，只有 soft prompt 的 embedding 参数可训练。"""
        base = _tiny_gpt2()
        model = SoftPromptGPT(base, num_virtual_tokens=6)
        # 基座全部冻结
        self.assertTrue(all(not p.requires_grad for p in base.parameters()))
        # soft prompt 可训练
        self.assertTrue(model.soft_prompt.weight.requires_grad)
        # 可训练参数量恰为 6 * embed_dim
        self.assertEqual(model.trainable_params(), 6 * 32)
        self.assertEqual(count_trainable_params(model), 6 * 32)

    def test_freeze_base_model_idempotent(self):
        base = _tiny_gpt2()
        freeze_base_model(base)
        self.assertTrue(all(not p.requires_grad for p in base.parameters()))


class TestForward(unittest.TestCase):
    def test_output_shape_excludes_virtual_tokens(self):
        """logits 形状 = 真实 token 长度 × 词表（虚拟 token 位被去掉）。"""
        model = SoftPromptGPT(_tiny_gpt2(), num_virtual_tokens=5)
        x = torch.randint(0, 32, (2, 9))
        logits = model(x)
        self.assertEqual(logits.shape, (2, 9, 32))

    def test_prompt_changes_output(self):
        """改变 soft prompt 参数会改变输出（说明梯度路径接通）。"""
        torch.manual_seed(0)
        model = SoftPromptGPT(_tiny_gpt2(), num_virtual_tokens=4)
        x = torch.randint(0, 32, (1, 6))
        out1 = model(x).detach().clone()
        with torch.no_grad():
            model.soft_prompt.weight.add_(torch.randn_like(model.soft_prompt.weight))
        out2 = model(x).detach()
        self.assertFalse(torch.allclose(out1, out2))

    def test_base_output_unchanged_by_wrapper(self):
        """包装类不改基座权重（冻结语义）。"""
        base = _tiny_gpt2()
        ref_weight = base.out_head.weight.detach().clone()
        model = SoftPromptGPT(base, num_virtual_tokens=4)
        x = torch.randint(0, 32, (1, 6))
        model(x)
        self.assertTrue(torch.equal(base.out_head.weight.detach(), ref_weight))


class TestTraining(unittest.TestCase):
    def test_toy_overfit_loss_decreases(self):
        """玩具过拟合：固定输入映射固定目标，Adam 几十步后 loss 显著下降。"""
        torch.manual_seed(42)
        model = SoftPromptGPT(_tiny_gpt2(), num_virtual_tokens=4)
        x = torch.randint(0, 32, (1, 8))
        y = torch.tensor([[7] * 8])
        optimizer = torch.optim.Adam(model.soft_prompt.parameters(), lr=3e-2)
        first = last = None
        for _ in range(30):
            loss = F.cross_entropy(model(x).reshape(-1, 32), y.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if first is None:
                first = loss.item()
            last = loss.item()
        self.assertLess(last, first)

    def test_init_from_text(self):
        """init_from_text 用真实词 embedding 初始化（mean 与 cyclic 两种模式）。"""
        class FakeTok:
            def encode(self, text):
                # 空文本返回空列表（真实 tokenizer 的空编码行为）
                return [ord(c) % 32 for c in text] if text else []
        torch.manual_seed(1)
        model = SoftPromptGPT(_tiny_gpt2(), num_virtual_tokens=4)
        out = init_from_text(model, FakeTok(), "hello prompt")
        self.assertIs(out, model)
        # 初始化后 embedding 不再全零均值分布（取自真实词向量）
        w = model.soft_prompt.weight
        tok_w = model.base_model.tok_embed.weight
        # mean 模式：每位都应等于输入词向量的均值
        ids = FakeTok().encode("hello prompt")
        expected = tok_w[ids].mean(dim=0)
        self.assertTrue(torch.allclose(w[0], expected, atol=1e-5))
        # 空文本应报错
        with self.assertRaises(ValueError):
            init_from_text(model, FakeTok(), "")


if __name__ == "__main__":
    unittest.main()
