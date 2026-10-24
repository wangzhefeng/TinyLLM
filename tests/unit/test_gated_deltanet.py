"""Gated DeltaNet（线性注意力，arXiv:2412.06464）单元测试。

覆盖：
1. recurrent 与 chunked 输出逐元素等价（核心验收点，atol 1e-4；
   含非整除块长、随机初始状态、块末状态等价）
2. 两种模式的形状 / 有限性
3. 因果前缀不变性（修改未来 token 不影响历史输出）
4. beta、gate ∈ (0, 1) 开区间；key 按头 L2 归一化
5. v=0 且 g<1 时状态范数随 t 严格衰减（门控遗忘语义）
6. 反向传播梯度有限
"""
import unittest

import torch

from layers.gated_delta_rule import GatedDeltaNet

# tiny 配置（离线、CPU）
D_MODEL, N_HEADS, DK, DV, T = 32, 2, 16, 16, 24


def _layer(**overrides):
    params = dict(d_model=D_MODEL, n_heads=N_HEADS, dk=DK, dv=DV)
    params.update(overrides)
    torch.manual_seed(0)
    return GatedDeltaNet(**params)


def _input(batch_size=2, num_tokens=T):
    torch.manual_seed(42)
    return torch.randn(batch_size, num_tokens, D_MODEL)


class TestRecurrentChunkedEquivalence(unittest.TestCase):
    """chunked 是逐 token 门控的精确并行形式，二者必须逐元素一致。"""

    def test_outputs_match_chunk_16(self):
        layer = _layer().eval()
        x = _input()
        with torch.no_grad():
            out_rec = layer(x, mode="recurrent")
            out_chk = layer(x, mode="chunk", chunk_size=16)
        self.assertTrue(
            torch.allclose(out_rec, out_chk, atol=1e-4),
            msg=f"max|diff|={(out_rec - out_chk).abs().max().item():.3e}",
        )

    def test_outputs_match_partial_chunk(self):
        """T=24 不被 5 整除（末块长 4），非整除情形也必须等价。"""
        layer = _layer().eval()
        x = _input()
        with torch.no_grad():
            out_rec = layer.forward_recurrent(x)
            out_chk = layer.forward_chunked(x, chunk_size=5)
        self.assertTrue(
            torch.allclose(out_rec, out_chk, atol=1e-4),
            msg=f"max|diff|={(out_rec - out_chk).abs().max().item():.3e}",
        )

    def test_outputs_match_with_initial_state(self):
        """携带随机初始状态（跨段续算场景）时仍等价。"""
        layer = _layer().eval()
        x = _input()
        S0 = torch.randn(2, N_HEADS, DK, DV)
        with torch.no_grad():
            out_rec = layer.forward_recurrent(x, initial_state=S0)
            out_chk = layer.forward_chunked(x, chunk_size=10, initial_state=S0)
        self.assertTrue(
            torch.allclose(out_rec, out_chk, atol=1e-4),
            msg=f"max|diff|={(out_rec - out_chk).abs().max().item():.3e}",
        )

    def test_block_boundary_states_match(self):
        """块边界处（t=15、t=23）的状态也必须与递推一致。"""
        layer = _layer().eval()
        x = _input()
        with torch.no_grad():
            _, states_rec = layer.forward_recurrent(x, return_states=True)
            _, states_chk = layer.forward_chunked(x, chunk_size=16, return_states=True)
        self.assertEqual(states_rec.shape, (2, N_HEADS, T, DK, DV))
        self.assertEqual(states_chk.shape, (2, N_HEADS, 2, DK, DV))
        for t_rec, c_chk in [(15, 0), (T - 1, 1)]:
            self.assertTrue(
                torch.allclose(states_rec[:, :, t_rec], states_chk[:, :, c_chk], atol=1e-4),
                msg=f"state at t={t_rec} mismatch",
            )


class TestShapeAndFiniteness(unittest.TestCase):
    def test_shapes_and_finite(self):
        layer = _layer().eval()
        x = _input()
        with torch.no_grad():
            for mode in ["recurrent", "chunk"]:
                out = layer(x, mode=mode, chunk_size=16)
                self.assertEqual(out.shape, (2, T, D_MODEL))
                self.assertTrue(torch.isfinite(out).all())

    def test_short_sequence(self):
        """T 小于块长（单块、且不整除）也能正常前向。"""
        layer = _layer().eval()
        x = _input(num_tokens=7)
        with torch.no_grad():
            out = layer(x, mode="chunk", chunk_size=16)
        self.assertEqual(out.shape, (2, 7, D_MODEL))
        self.assertTrue(torch.isfinite(out).all())


class TestCausality(unittest.TestCase):
    def test_future_tokens_do_not_affect_past_outputs(self):
        """状态式更新天然因果：改动 t>=12 的 token，t<12 的输出不变。"""
        layer = _layer().eval()
        torch.manual_seed(7)
        x1 = torch.randn(2, T, D_MODEL)
        x2 = x1.clone()
        x2[:, 12:, :] = torch.randn_like(x2[:, 12:, :])      # 只改未来
        with torch.no_grad():
            for mode in ["recurrent", "chunk"]:
                out1 = layer(x1, mode=mode, chunk_size=16)
                out2 = layer(x2, mode=mode, chunk_size=16)
                self.assertTrue(
                    torch.allclose(out1[:, :12], out2[:, :12], atol=1e-6),
                    msg=f"mode={mode} violates causality",
                )


class TestGating(unittest.TestCase):
    def test_beta_gate_in_open_unit_interval_and_key_normalized(self):
        layer = _layer().eval()
        x = _input()
        _, k, _, beta, gate = layer._project(x)
        for name, val in [("beta", beta), ("gate", gate)]:
            self.assertTrue((val > 0).all(), msg=f"{name} must be > 0")
            self.assertTrue((val < 1).all(), msg=f"{name} must be < 1")
        # key 按头 L2 归一化
        self.assertTrue(
            torch.allclose(k.norm(dim=-1), torch.ones_like(k.norm(dim=-1)), atol=1e-5)
        )

    def test_state_norm_decays_when_v_zero_and_gate_lt_one(self):
        """v≡0、g<1：S_t = g(I−β k kᵀ)S_{t−1} 为收缩映射，范数严格递减。"""
        layer = _layer().eval()
        with torch.no_grad():
            layer.W_value.weight.zero_()       # v ≡ 0
            layer.W_gate.bias.fill_(-1.0)      # g ≈ σ(−1) ≈ 0.27，显著衰减
        x = _input()
        S0 = torch.randn(2, N_HEADS, DK, DV)
        with torch.no_grad():
            _, states = layer.forward_recurrent(x, initial_state=S0, return_states=True)
        norms = states.norm(dim=(-2, -1))      # (B, H, T)
        self.assertTrue((norms > 0).all())
        # 每一步严格衰减
        self.assertTrue(
            (norms[:, :, 1:] < norms[:, :, :-1]).all(),
            msg=f"norms not strictly decreasing: {norms[0, 0].tolist()}",
        )
        # 24 步 × g≈0.27 后应远小于初值
        self.assertTrue((norms[:, :, -1] < 0.1 * norms[:, :, 0]).all())


class TestBackward(unittest.TestCase):
    def test_gradients_finite_both_modes(self):
        for mode in ["recurrent", "chunk"]:
            layer = _layer()
            x = _input().requires_grad_(True)
            out = layer(x, mode=mode, chunk_size=16)
            out.pow(2).sum().backward()
            grads = [p.grad for p in layer.parameters()]
            self.assertTrue(all(g is not None for g in grads), msg=f"mode={mode}")
            for (name, _), g in zip(layer.named_parameters(), grads):
                self.assertTrue(
                    torch.isfinite(g).all(), msg=f"mode={mode}, grad of {name} not finite"
                )
            self.assertTrue(torch.isfinite(x.grad).all(), msg=f"mode={mode}")


if __name__ == "__main__":
    unittest.main()
