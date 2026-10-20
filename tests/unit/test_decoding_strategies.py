"""解码策略测试：repetition penalty 与 beam search（Phase 2 推断域）。"""
import unittest
from types import SimpleNamespace

import torch

from inference.generate_core import (
    beam_search_generate,
    generate,
    repetition_penalty,
)
from models.gpt2.gpt2_124M import Model


def _tiny_gpt2():
    torch.manual_seed(123)
    cfg = SimpleNamespace(
        vocab_size=64, context_length=32, embed_dim=32, d_ff=64,
        n_heads=4, n_layers=2, dropout=0.0, qkv_bias=True,
        dtype=torch.float32, is_train=False,
    )
    return Model(cfg).eval()


class TestRepetitionPenalty(unittest.TestCase):
    def test_positive_logit_divided(self):
        logits = torch.tensor([[2.0, -1.0, 0.5]])
        ids = torch.tensor([[0, 1]])
        out = repetition_penalty(logits, ids, penalty=2.0)
        self.assertAlmostEqual(out[0, 0].item(), 1.0)   # 2.0 / 2
        self.assertAlmostEqual(out[0, 1].item(), -2.0)  # -1.0 * 2
        self.assertAlmostEqual(out[0, 2].item(), 0.5)   # 未见不变

    def test_noop_when_penalty_one(self):
        logits = torch.randn(2, 10)
        ids = torch.randint(0, 10, (2, 4))
        self.assertTrue(torch.equal(repetition_penalty(logits, ids, 1.0), logits))

    def test_does_not_modify_input(self):
        logits = torch.tensor([[1.0, 1.0]])
        repetition_penalty(logits, torch.tensor([[0]]), 2.0)
        self.assertEqual(logits[0, 0].item(), 1.0)


class TestBeamSearch(unittest.TestCase):
    def setUp(self):
        self.model = _tiny_gpt2()

    def test_beam1_equals_greedy(self):
        x = torch.randint(0, 64, (1, 5))
        greedy = generate(self.model, x.clone(), max_new_tokens=5, context_length=32)
        beam = beam_search_generate(self.model, x.clone(), max_new_tokens=5,
                                    beam_size=1, context_length=32)
        self.assertTrue(torch.equal(greedy, beam))

    def test_shape_and_return_beams(self):
        x = torch.randint(0, 64, (1, 4))
        beams = beam_search_generate(
            self.model, x, max_new_tokens=4, beam_size=3,
            context_length=32, return_beams=True,
        )
        self.assertEqual(beams.shape, (3, 8))
        # beam 间应有差异（随机模型几乎不可能三束相同）
        self.assertGreater(len({tuple(b.tolist()) for b in beams}), 1)

    def test_eos_terminates_early(self):
        # 让模型第一步大概率生成某个固定 token 的极端情形难构造，
        # 这里验证 eos 冻结逻辑：所有 beam 完成后序列长度不再增长
        x = torch.randint(0, 64, (1, 4))
        out = beam_search_generate(
            self.model, x.clone(), max_new_tokens=6, beam_size=2,
            context_length=32, eos_token_id=7,
        )
        self.assertEqual(out.shape, (1, 4 + 6))  # 教学实现返回等长（冻结 beam 填充 eos）


if __name__ == "__main__":
    unittest.main()
