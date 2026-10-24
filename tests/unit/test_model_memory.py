"""model_memory 参数统计测试：weight tying 检测与唯一参数量扣除（共享层正确性修复）。"""
import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn

from utils.model_memory import model_memory_size, model_param_counts


class _GPT2Like(nn.Module):
    """GPT2 结构的最小替身：tok_embed + 独立 out_head，可选 tying。"""

    def __init__(self, vocab_size=50, embed_dim=8, tied=False):
        super().__init__()
        self.tok_embed = nn.Embedding(vocab_size, embed_dim)
        self.out_head = nn.Linear(embed_dim, vocab_size, bias=False)
        if tied:
            self.out_head.weight = self.tok_embed.weight


class TestModelParamCounts(unittest.TestCase):
    def test_untied_no_deduction(self):
        """未 tying（GPT2 情形）：不扣 embedding，唯一参数量 == 注册参数量。"""
        vocab, dim = 50, 8
        model = _GPT2Like(vocab, dim, tied=False)
        counts = model_param_counts(model)
        expected = vocab * dim * 2  # embedding + out_head 各自独立
        self.assertFalse(counts["weight_tied"])
        self.assertEqual(counts["total_params"], expected)
        self.assertEqual(counts["total_params_normalized"], expected)

    def test_tied_deducts_embedding_once(self):
        """实际 tying：唯一参数量 = 注册参数量 - 一份 embedding。"""
        vocab, dim = 50, 8
        model = _GPT2Like(vocab, dim, tied=True)
        counts = model_param_counts(model)
        self.assertTrue(counts["weight_tied"])
        self.assertEqual(counts["total_params"], vocab * dim * 2)
        self.assertEqual(counts["total_params_normalized"], vocab * dim)
        # 唯一参数量与去重遍历结果一致
        unique = sum(p.numel() for p in model.parameters())
        self.assertEqual(counts["total_params_normalized"], unique)

    def test_real_gpt2_model_not_tied(self):
        """真实 GPT2 模型（tok_embed 与 out_head 独立）：统计不扣 embedding。"""
        from models.gpt2.gpt2_124M import Model

        cfg = SimpleNamespace(
            vocab_size=64, embed_dim=16, context_length=8, dropout=0.0,
            n_layers=1, n_heads=2, qkv_bias=False, d_ff=32, dtype=torch.float32,
        )
        model = Model(cfg)
        counts = model_param_counts(model)
        self.assertFalse(counts["weight_tied"])
        total = sum(p.numel() for p in model.parameters())
        self.assertEqual(counts["total_params"], total)
        self.assertEqual(counts["total_params_normalized"], total)

    def test_memory_uses_unique_params(self):
        """tying 时显存估算按物理存储（embedding 只占一份）。"""
        vocab, dim = 50, 8
        model = _GPT2Like(vocab, dim, tied=True)
        gb = model_memory_size(model, input_dtype=torch.float32, verbose=False)
        counts = model_param_counts(model)
        expected = (
            counts["total_params_normalized"]
            + counts["total_grads"]
            + counts["total_buffers"]
        ) * 4 / (1024 ** 3)
        self.assertAlmostEqual(gb, expected, places=12)


if __name__ == "__main__":
    unittest.main()
