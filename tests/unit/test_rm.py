"""Reward Model（RLHF 一阶段）测试：价值头 + Bradley-Terry 成对损失。"""
import unittest
from types import SimpleNamespace

import torch

from models.gpt2.gpt2_124M import Model
from post_training.preference.rm import (
    RewardModel,
    bradley_terry_loss,
    rm_pairwise_step,
)


def _tiny_cfg():
    return SimpleNamespace(
        vocab_size=100,
        context_length=32,
        embed_dim=32,
        d_ff=64,
        n_heads=4,
        n_layers=2,
        dropout=0.0,
        qkv_bias=True,
        dtype=torch.float32,
    )


def _build_rm():
    torch.manual_seed(0)
    cfg = _tiny_cfg()
    return RewardModel(Model(cfg), cfg)


class TestBradleyTerryLoss(unittest.TestCase):
    def test_loss_decreases_with_margin(self):
        """margin 越大损失越小；0 margin 时 loss = ln2。"""
        r = torch.zeros(4)
        self.assertAlmostEqual(float(bradley_terry_loss(r, r)), 0.6931, places=3)
        small = bradley_terry_loss(torch.tensor([1.0]), torch.tensor([0.0]))
        large = bradley_terry_loss(torch.tensor([4.0]), torch.tensor([0.0]))
        self.assertLess(float(large), float(small))

    def test_gradient_direction(self):
        """对 r_chosen 的梯度为负：增大 chosen reward 应降低损失。"""
        r_c = torch.zeros(1, requires_grad=True)
        r_r = torch.zeros(1)
        loss = bradley_terry_loss(r_c, r_r)
        loss.backward()
        self.assertLess(float(r_c.grad), 0.0)


class TestRewardModel(unittest.TestCase):
    def test_token_rewards_shape(self):
        rm = _build_rm().eval()
        ids = torch.randint(0, 100, (2, 16))
        with torch.no_grad():
            rewards = rm.token_rewards(ids)
        self.assertEqual(rewards.shape, (2, 16))
        self.assertTrue(torch.isfinite(rewards).all())

    def test_sequence_reward_takes_last_non_pad(self):
        """右 padding：序列得分等于最后一个非 pad token 的逐 token reward。"""
        rm = _build_rm().eval()
        ids = torch.tensor([[3, 5, 7, 99, 99]])
        with torch.no_grad():
            token_r = rm.token_rewards(ids)
            seq_r = rm(ids, pad_token_id=99)
        torch.testing.assert_close(seq_r[0], token_r[0, 2])

    def test_v_head_only_adds_scalar_head(self):
        """v_head 只有 embed_dim+1 个参数；backbone 参数原样保留。"""
        rm = _build_rm()
        self.assertEqual(rm.v_head.weight.shape, (1, 32))
        self.assertEqual(rm.v_head.bias.shape, (1,))
        self.assertIn("v_head.weight", dict(rm.named_parameters()))
        self.assertIn("backbone.tok_embed.weight", dict(rm.named_parameters()))

    def test_pairwise_step_overfits_single_batch(self):
        """同一 batch 连训两步：BT 损失下降、margin 上升。"""
        torch.manual_seed(1)
        rm = _build_rm()
        optimizer = torch.optim.AdamW(rm.parameters(), lr=5e-2)
        chosen = torch.randint(0, 100, (2, 12))
        rejected = torch.randint(0, 100, (2, 12))
        loss1, margin1 = rm_pairwise_step(rm, optimizer, chosen, rejected)
        loss2, margin2 = rm_pairwise_step(rm, optimizer, chosen, rejected)
        self.assertLess(loss2, loss1)
        self.assertGreater(margin2, margin1)

    def test_backbone_hidden_states_optin(self):
        """GPT-2 forward 默认只返回 logits；opt-in 返回 (logits, hidden)。"""
        model = Model(_tiny_cfg()).eval()
        ids = torch.randint(0, 100, (1, 8))
        with torch.no_grad():
            out = model(ids)
            logits, hidden = model(ids, output_hidden_states=True)
        self.assertIsInstance(out, torch.Tensor)
        self.assertEqual(logits.shape, (1, 8, 100))
        self.assertEqual(hidden.shape, (1, 8, 32))


if __name__ == "__main__":
    unittest.main()
