# -*- coding: utf-8 -*-

# ***************************************************
# * File        : gated_delta_rule.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2026-10-24
# * Version     : 1.0.102400
# * Description : Gated DeltaNet 教学实现（线性注意力 / 快速权重记忆）
# *               对应 LLMs-from-scratch ch04/08 bonus，arXiv:2412.06464。
# *               每头维护状态矩阵 S: (dk, dv)，递推（ground truth）：
# *                 S_t = g_t·(I − β_t·k_t k_tᵀ)·S_{t−1} + β_t·k_t v_tᵀ
# *                 o_t = q_tᵀ·S_t（查询作用于更新后的状态）
# *               forward_chunked 为逐 token 门控的**精确**块内并行形式
# *               （论文 Algorithm 2 的 WY 表示），与 recurrent 逐元素等价，
# *               未采用"块内常数门控"量化折衷。
# * Link        : https://arxiv.org/abs/2412.06464
# *               https://github.com/rasbt/LLMs-from-scratch
# * Requirement : torch
# ***************************************************

__all__ = [
    "GatedDeltaNet",
]

# python libraries
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


class GatedDeltaNet(nn.Module):
    """
    Gated DeltaNet：带门控的 delta rule 线性注意力（arXiv:2412.06464）。

    每头维护一个"快速权重"记忆矩阵 S: (dk, dv)，逐 token 递推::

        S_t = g_t · (I − β_t · k_t k_tᵀ) · S_{t−1} + β_t · k_t v_tᵀ
        o_t = q_tᵀ · S_t        # 查询作用于"更新后"的状态

    - delta rule：写入新关联 (k_t, v_t) 前，先用 (I − β_t k_t k_tᵀ) 删除
      key 方向上已有的旧关联，是比朴素线性注意力更强的关键；
    - 门控 g_t ∈ (0,1)：对记忆做指数级遗忘，长序列下近似 softmax
      attention 的衰减效应。论文形式为 g = exp(−λ·σ(α·x))；教学简化为
      直接 sigmoid，即 g = σ(W_gate·x) ∈ (0,1)，仍保持衰减语义。

    投影（输入 x: (B, T, d_model)）：
      - W_query : d_model -> H·dk（不缩放，保持 o = qᵀS 的教科书形式）
      - W_key   : d_model -> H·dk，输出按头 L2 归一化（论文做法，
                  保证 (I − β k kᵀ) 是非扩张算子，训练稳定）
      - W_value : d_model -> H·dv
      - W_beta  : d_model -> H，每头每 token 一个标量，sigmoid 得 β ∈ (0,1)
      - W_gate  : d_model -> H，每头每 token 一个标量，sigmoid 得 g ∈ (0,1)
      - out_proj: H·dv -> d_model

    chunked 并行形式（精确，与 recurrent 等价）：
    设块内下标从 1 到 C，Γ_i = ∏_{l≤i} g_l 为块内累积门控，
    R_{ij} = Γ_i/Γ_j (j ≤ i)，WY 表示 T = (I + diag(β)·tril(K Kᵀ, −1))⁻¹
    （单位下三角，恒可逆），则::

        a    = tril(Q Kᵀ) @ T            # 经过块内 delta 纠正的注意力 logits
        û    = Tᵀ @ K                    # 被"后续 token 的 delta"修正过的 keys
        S_kv = K @ S₀                    # 初始状态在块内 keys 上的读出
        o    = Γ ⊙ (Q·S₀ − (a·β)·S_kv) + (R ⊙ (a·β))·V
        S_out= Γ_C·S₀ + ûᵀ·[β ⊙ (R_{C,·}·V − Γ_C·S_kv)]

    数值细节：C×C 三角矩阵求逆与累积门控 Γ/R 在 float64 中计算后转回输入
    dtype，避免长块连乘 g 的精度损失；块与块之间仅传递状态 S，天然因果，
    无需因果掩码。

    计算复杂度：recurrent 为 O(T·dk·dv)（串行），chunked 为
    O(T·C·(dk+dv) + T/C·dk·dv·C)（块内矩阵乘并行）。
    """

    def __init__(self, d_model: int, n_heads: int, dk: int = None, dv: int = None):
        super().__init__()

        assert n_heads >= 1, "n_heads must be >= 1"
        self.d_model = d_model
        self.n_heads = n_heads
        self.dk = dk if dk is not None else d_model // n_heads
        self.dv = dv if dv is not None else d_model // n_heads

        # 投影层
        self.W_query = nn.Linear(d_model, self.n_heads * self.dk, bias=False)
        self.W_key = nn.Linear(d_model, self.n_heads * self.dk, bias=False)
        self.W_value = nn.Linear(d_model, self.n_heads * self.dv, bias=False)
        self.W_beta = nn.Linear(d_model, self.n_heads, bias=True)
        self.W_gate = nn.Linear(d_model, self.n_heads, bias=True)
        self.out_proj = nn.Linear(self.n_heads * self.dv, d_model, bias=False)

    def _project(self, x):
        """
        输入投影 + 分头。

        Args:
            x: (batch_size, num_tokens, d_model)

        Returns:
            q: (B, H, T, dk)
            k: (B, H, T, dk)，按头 L2 归一化
            v: (B, H, T, dv)
            beta: (B, H, T)，sigmoid，∈ (0, 1)
            gate: (B, H, T)，sigmoid，∈ (0, 1)
        """
        batch_size, num_tokens, _ = x.shape
        # shape: (B, T, D) -> (B, H, T, ·)
        q = self.W_query(x).view(batch_size, num_tokens, self.n_heads, self.dk).transpose(1, 2)
        k = self.W_key(x).view(batch_size, num_tokens, self.n_heads, self.dk).transpose(1, 2)
        v = self.W_value(x).view(batch_size, num_tokens, self.n_heads, self.dv).transpose(1, 2)
        # key L2 归一化（论文做法）
        k = F.normalize(k, dim=-1, eps=1e-6)
        # 每头标量门控
        beta = torch.sigmoid(self.W_beta(x))  # (B, T, H)
        gate = torch.sigmoid(self.W_gate(x))  # (B, T, H)
        beta = beta.view(batch_size, num_tokens, self.n_heads).transpose(1, 2)
        gate = gate.view(batch_size, num_tokens, self.n_heads).transpose(1, 2)

        return q, k, v, beta, gate

    def _init_state(self, x, initial_state):
        if initial_state is None:
            return x.new_zeros(x.shape[0], self.n_heads, self.dk, self.dv)
        return initial_state.clone()

    def forward_recurrent(self, x, initial_state=None, return_states=False):
        """
        逐 token 递推（ground truth 参考实现）。

        Args:
            x: (B, T, d_model)
            initial_state: (B, H, dk, dv) 初始记忆状态，None 则置零
            return_states: 为 True 时额外返回每步状态 (B, H, T, dk, dv)

        Returns:
            out: (B, T, d_model)；return_states=True 时为 (out, states)
        """
        batch_size, num_tokens, _ = x.shape
        q, k, v, beta, gate = self._project(x)

        S = self._init_state(x, initial_state)
        outs, states = [], []
        for t in range(num_tokens):
            q_t = q[:, :, t:t + 1, :]                                        # (B, H, 1, dk)
            k_t = k[:, :, t:t + 1, :]                                        # (B, H, 1, dk)
            v_t = v[:, :, t:t + 1, :]                                        # (B, H, 1, dv)
            beta_t = beta[:, :, t:t + 1].unsqueeze(-1)                       # (B, H, 1, 1)
            g_t = gate[:, :, t:t + 1].unsqueeze(-1)                          # (B, H, 1, 1)
            # delta rule 状态更新：
            # S_t = g_t·(I − β_t·k_t k_tᵀ)·S_{t−1} + β_t·k_t v_tᵀ
            kS = torch.matmul(k_t, S)                                      # (B, H, 1, dv) = k_tᵀ S
            S = g_t * (S - beta_t * torch.matmul(k_t.transpose(-1, -2), kS)) \
                + beta_t * torch.matmul(k_t.transpose(-1, -2), v_t)
            if return_states:
                states.append(S)
            # o_t = q_tᵀ S_t（更新后的状态）
            outs.append(torch.matmul(q_t, S))                                # (B, H, 1, dv)

        # 合并头 + 输出投影
        out = torch.cat(outs, dim=2).transpose(1, 2).reshape(batch_size, num_tokens, -1)
        out = self.out_proj(out)

        if return_states:
            return out, torch.stack(states, dim=2)                           # (B, H, T, dk, dv)
        return out

    def forward_chunked(self, x, chunk_size: int = 16, initial_state=None, return_states=False):
        """
        块内并行的精确形式（与 forward_recurrent 逐元素等价）。

        参照论文 Algorithm 2 的 WY 表示：块内用矩阵乘并行完成全部
        token 间交互，块间只传递状态 S。逐 token 门控 g_t 被精确处理为
        块内累积门控 Γ 及其比值 R（未做"块内常数门控"量化折衷）。

        Args:
            x: (B, T, d_model)
            chunk_size: 块长 C（T 不整除时末块自动缩短）
            initial_state: (B, H, dk, dv) 初始记忆状态，None 则置零
            return_states: 为 True 时额外返回每块末状态 (B, H, n_chunks, dk, dv)

        Returns:
            out: (B, T, d_model)；return_states=True 时为 (out, states)
        """
        batch_size, num_tokens, _ = x.shape
        q, k, v, beta, gate = self._project(x)

        S = self._init_state(x, initial_state)
        outs, states = [], []
        for c_start in range(0, num_tokens, chunk_size):
            c_end = min(c_start + chunk_size, num_tokens)
            C = c_end - c_start
            Qc = q[:, :, c_start:c_end, :]                                   # (B, H, C, dk)
            Kc = k[:, :, c_start:c_end, :]                                   # (B, H, C, dk)
            Vc = v[:, :, c_start:c_end, :]                                   # (B, H, C, dv)
            beta_c = beta[:, :, c_start:c_end]                               # (B, H, C)
            gate_c = gate[:, :, c_start:c_end]                               # (B, H, C)

            # ---- 数值敏感量用 float64 计算，再转回输入 dtype ----
            dtype = Qc.dtype
            # 块内累积门控 Γ_i = ∏_{l≤i} g_l，及比值 R_{ij} = Γ_i/Γ_j (j≤i)。
            # 用 log-cumsum-exp 之外的直接 cumprod 即可（C 较小），
            # float64 下 1/Γ_j 的动态范围可承受。
            gamma_64 = torch.cumprod(gate_c.double(), dim=-1)                # (B, H, C)
            R_64 = gamma_64.unsqueeze(-1) / gamma_64.unsqueeze(-2)           # R_{ij} = Γ_i/Γ_j
            R_64 = torch.tril(R_64, diagonal=0)                              # 下三角（含对角=1）
            gamma = gamma_64.to(dtype)
            R = R_64.to(dtype)

            # ---- WY 表示：T = (I + diag(β)·tril(K Kᵀ, −1))⁻¹（单位下三角） ----
            gram = torch.matmul(Kc.double(), Kc.double().transpose(-1, -2))  # (B, H, C, C)
            eye = torch.eye(C, device=x.device, dtype=torch.float64)
            T_inv = torch.linalg.inv(eye + beta_c.double().unsqueeze(-1) * torch.tril(gram, diagonal=-1))
            T_inv = T_inv.to(dtype)

            # ---- 块内输出与块末状态（精确并行形式） ----
            # a_{ij} = q_iᵀ·ẑ_j^{(i)}：被后续 delta 纠正后的块内注意力 logits，
            # 因果性由 tril(QKᵀ) 与 T_inv 的下三角性共同保证
            a = torch.matmul(torch.tril(torch.matmul(Qc, Kc.transpose(-1, -2)), diagonal=0), T_inv)
            a_beta = a * beta_c.unsqueeze(-2)                                # (B, H, C, C)，j 维乘 β_j
            S_kv = torch.matmul(Kc, S)                                       # (B, H, C, dv) = K·S₀
            gamma_c_last = gamma[..., -1:]                                   # (B, H, 1) = Γ_C
            # o = Γ_i·(q_iᵀS₀ − Σ_j β_j a_ij·(k_jᵀS₀)) + Σ_j (Γ_i/Γ_j)·β_j a_ij·v_jᵀ
            o = gamma.unsqueeze(-1) * (torch.matmul(Qc, S) - torch.matmul(a_beta, S_kv)) \
                + torch.matmul(R * a_beta, Vc)
            # S_out = Γ_C·S₀ + ûᵀ·[β ⊙ (R_{C,·}·V − Γ_C·S_kv)]，û = TᵀK
            u_hat = torch.matmul(T_inv.transpose(-1, -2), Kc)                # (B, H, C, dk)
            R_last = R[..., -1, :]                                           # (B, H, C)
            delta_state = torch.matmul(
                u_hat.transpose(-1, -2),                                     # (B, H, dk, C)
                beta_c.unsqueeze(-1) * (R_last.unsqueeze(-1) * Vc - gamma_c_last.unsqueeze(-1) * S_kv),
            )
            S = gamma_c_last.unsqueeze(-1) * S + delta_state

            outs.append(o)
            if return_states:
                states.append(S)

        # 合并块 + 头，输出投影
        out = torch.cat(outs, dim=2).transpose(1, 2).reshape(batch_size, num_tokens, -1)
        out = self.out_proj(out)

        if return_states:
            return out, torch.stack(states, dim=2)                           # (B, H, n_chunks, dk, dv)
        return out

    def forward(self, x, mode: str = "chunk", chunk_size: int = 16):
        """
        前向分派。状态式更新天然因果，无需因果掩码。

        Args:
            x: (B, T, d_model)
            mode: "recurrent"（逐 token 递推）或 "chunk"（块内并行）
            chunk_size: chunk 模式的块长

        Returns:
            out: (B, T, d_model)
        """
        if mode == "recurrent":
            return self.forward_recurrent(x)
        elif mode == "chunk":
            return self.forward_chunked(x, chunk_size=chunk_size)
        raise ValueError(f"Unknown mode: {mode!r}, expected 'recurrent' or 'chunk'")


# 测试代码 main 函数
def main():
    torch.manual_seed(0)
    layer = GatedDeltaNet(d_model=32, n_heads=2, dk=16, dv=16)

    x = torch.randn(2, 24, 32)
    out_rec = layer(x, mode="recurrent")
    out_chk = layer(x, mode="chunk", chunk_size=16)
    print(f"input: {tuple(x.shape)}")
    print(f"recurrent out: {tuple(out_rec.shape)}, chunked out: {tuple(out_chk.shape)}")
    print(f"两路径最大绝对差: {(out_rec - out_chk).abs().max().item():.3e}")

    # 门控衰减演示：v=0、g<1 时状态范数逐 token 递减
    with torch.no_grad():
        layer.W_value.weight.zero_()      # v ≡ 0
        layer.W_gate.bias.fill_(-1.0)     # g ≈ σ(−1) ≈ 0.27
    S0 = torch.randn(2, 2, 16, 16)
    _, states = layer.forward_recurrent(x, initial_state=S0, return_states=True)
    norms = states.norm(dim=(-2, -1))     # (B, H, T)
    print(f"状态范数衰减: {norms[0, 0, :5].tolist()} ... -> {norms[0, 0, -1].item():.3e}")

if __name__ == "__main__":
    main()
