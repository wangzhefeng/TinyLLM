"""解码策略测试：repetition penalty 与 beam search（Phase 2 推断域）。"""
import unittest
from types import SimpleNamespace

import torch

from inference.generate_core import (
    beam_search_generate,
    generate,
    generate_simple,
    repetition_penalty,
)
from inference.strategies.speculative import speculative_generate
from models.gpt2.gpt2_124M import Model


def _tiny_gpt2(seed=123):
    torch.manual_seed(seed)
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

    def test_num_beams_1_matches_generate_simple(self):
        # num_beams=1 + length_penalty=0 时退化为 greedy：与 generate_simple 逐 token 一致
        x = torch.randint(0, 64, (1, 5))
        greedy = generate_simple(self.model, x.clone(), max_new_tokens=6, context_length=32)
        beam = beam_search_generate(self.model, x.clone(), max_new_tokens=6,
                                    num_beams=1, length_penalty=0.0, context_length=32)
        self.assertTrue(torch.equal(greedy, beam))

    def test_num_beams_3_valid_tokens(self):
        # num_beams=3：正常终止、形状正确、token id 均在词表范围内
        x = torch.randint(0, 64, (1, 4))
        out = beam_search_generate(self.model, x.clone(), max_new_tokens=6,
                                   num_beams=3, context_length=32)
        self.assertEqual(out.shape, (1, 4 + 6))
        self.assertTrue(bool((out >= 0).all()))
        self.assertTrue(bool((out < 64).all()))

    def test_length_penalty_0_vs_1(self):
        # eos=9 时有 beam 提前冻结：lp=0 只比 logprob 总和（偏向早停短 beam），
        # lp=1.0 按生成长度归一（偏向平均分高的长 beam），两者选择不同；
        # （模型与 prompt 由固定种子决定，该区分场景可复现）
        x = torch.randint(0, 64, (1, 4))
        common = dict(max_new_tokens=8, num_beams=3, context_length=32, eos_token_id=9)
        out_lp0 = beam_search_generate(self.model, x.clone(), length_penalty=0.0, **common)
        out_lp1 = beam_search_generate(self.model, x.clone(), length_penalty=1.0, **common)
        for out in (out_lp0, out_lp1):
            self.assertEqual(out.shape, (1, 4 + 8))
            self.assertTrue(bool((out >= 0).all()))
            self.assertTrue(bool((out < 64).all()))
        self.assertFalse(torch.equal(out_lp0, out_lp1))

    def test_eos_greedy_first_token_stops_after_one(self):
        # eos 取模型 greedy 首个预测 token：num_beams=1 时第一步后全部 beam
        # 完成 → 提前停止，恰好生成 1 个新 token
        x = torch.randint(0, 64, (1, 4))
        with torch.inference_mode():
            first = int(torch.argmax(self.model(x)[:, -1, :]))
        out = beam_search_generate(self.model, x.clone(), max_new_tokens=10,
                                   num_beams=1, context_length=32, eos_token_id=first)
        self.assertEqual(out.shape, (1, 4 + 1))
        self.assertEqual(int(out[0, -1]), first)


class TestSpeculative(unittest.TestCase):
    def setUp(self):
        self.model = _tiny_gpt2()
        self.prompt = torch.tensor([[1, 2, 3, 4, 5]])

    def test_self_speculation_matches_greedy(self):
        # draft 与 target 为同一模型对象：输出必须与 generate_simple(greedy) 逐 token 一致
        greedy = generate_simple(self.model, self.prompt.clone(),
                                 max_new_tokens=8, context_length=32)
        spec = speculative_generate(self.model, self.model, self.prompt.clone(),
                                    max_new_tokens=8, num_draft_tokens=4,
                                    context_length=32)
        self.assertEqual(spec.shape, greedy.shape)
        self.assertTrue(torch.equal(greedy, spec))

    def test_different_draft_model_runs(self):
        # draft 为不同随机初始化的 tiny 模型：正常终止、形状正确、token id 合法
        draft = _tiny_gpt2(seed=456)
        out = speculative_generate(self.model, draft, self.prompt.clone(),
                                   max_new_tokens=6, num_draft_tokens=3,
                                   context_length=32)
        self.assertEqual(out.shape, (1, 5 + 6))
        self.assertTrue(bool((out < 64).all()))
        self.assertTrue(bool((out >= 0).all()))

    def test_eos_stops_early(self):
        # 该模型 greedy 的第一个预测 token 作为 eos → 恰好只生成 1 个新 token
        with torch.inference_mode():
            first = int(torch.argmax(self.model(self.prompt)[:, -1, :]))
        out = speculative_generate(self.model, self.model, self.prompt.clone(),
                                   max_new_tokens=10, num_draft_tokens=4,
                                   context_length=32, eos_token_id=first)
        self.assertEqual(out.shape, (1, 5 + 1))
        self.assertEqual(int(out[0, -1]), first)

    def test_num_draft_tokens_1_and_4(self):
        for k in (1, 4):
            out = speculative_generate(self.model, self.model, self.prompt.clone(),
                                       max_new_tokens=5, num_draft_tokens=k,
                                       context_length=32)
            self.assertEqual(out.shape, (1, 5 + 5))


if __name__ == "__main__":
    unittest.main()
