# -*- coding: utf-8 -*-

# ***************************************************
# * File        : gemma4.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2026-10-24
# * Version     : 1.0.102421
# * Description : Gemma 4（E2B/E4B）text-only 教学实现。
# *               架构已于 2026-10-24 对照官方材料核实：
# *               google/gemma-4-e2b 与 google/gemma-4-e4b 的 config.json，
# *               以及 transformers 5.5.0.dev0 的 configuration_gemma4.py /
# *               modeling_gemma4.py / modeling_rope_utils.py。
# *
# *               相对 Gemma 3 的演进点（均已核实）：
# *               1. local:global 交错比参数化：E2B 为 4:1（35 层）、E4B 为
# *                  5:1（42 层），transformers 默认 5:1，末层强制 full_attention
# *                  （见 layer_pattern）；
# *               2. 双 RoPE：sliding 层 theta=1e4 全旋转；full 层 theta=1e6 的
# *                  "proportional" partial RoPE（partial_rotary_factor=0.25，
# *                  频率指数仍按完整 head_dim 归一，未旋转频率对置 0，
# *                  等价于 cos=1/sin=0 的恒等旋转，见
# *                  precompute_proportional_rope_params）；
# *               3. full 层使用独立 head_dim（global_head_dim=512，sliding 层
# *                  256），通过逐层 cfg 复用 TransformerBlockGemma3 实现；
# *               4. q/k RMSNorm + attention scaling=1.0（query_pre_attn_scalar=1）；
# *               5. tie_word_embeddings + final_logit_softcapping=30（tanh 软限幅）；
# *               6. hidden_activation=gelu_pytorch_tanh（同 Gemma 3 FFN）。
# *
# *               官方存在、教学版暂未实现的效率特性（不改变教学主线，
# *               参数保留在 config/gemma4/model_cfgs.py 备查）：
# *               num_kv_shared_layers 跨层 KV 共享（末 20/18 层复用同类型层的
# *               K/V）、use_double_wide_mlp（仅 KV 共享层 FFN 加宽一倍）、
# *               PLE per-layer embeddings、attention v_norm、MoE block。
# * Link        : https://huggingface.co/google/gemma-4-e2b
# * Requirement : torch
# ***************************************************

__all__ = [
    "Model",
    "layer_pattern",
    "precompute_proportional_rope_params",
]

# python libraries
from pathlib import Path
import copy
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn

from layers.transformer_block import TransformerBlockGemma3
from layers.normalization.rms_norm import RMSNorm_Gemma3
from layers.position_encoding.RoPE import precompute_rope_params

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


def layer_pattern(n_layers: int, local_per_global: int = 5):
    """
    生成 Gemma 4 的 layer_types 交错序列。

    官方规则（configuration_gemma4.py）：周期为 ``local_per_global + 1``，
    每周期 ``local_per_global`` 个 sliding_attention + 1 个 full_attention
    （full 落在周期末），且最后一层强制 full_attention。

    Args:
        n_layers: 层数
        local_per_global: 每个全局注意力层对应的滑动注意力层数，
            即 local:global 交错比。E2B 官方为 4（4:1），E4B 与 transformers
            默认为 5（5:1）

    Returns:
        list[str]: 长度为 ``n_layers`` 的 layer_types
    """
    period = local_per_global + 1
    pattern = [
        "sliding_attention" if (i + 1) % period else "full_attention"
        for i in range(n_layers)
    ]
    pattern[-1] = "full_attention"
    return pattern


def precompute_proportional_rope_params(head_dim, theta_base=1_000_000.0,
                                        partial_rotary_factor=1.0,
                                        context_length=4096,
                                        dtype=torch.float32):
    """
    Gemma 4 "proportional" partial RoPE 的 cos/sin 预计算。

    对照 transformers ``_compute_proportional_rope_parameters`` +
    ``Gemma4TextRotaryEmbedding.forward``：

    - 旋转频率对数 ``rope_angles = int(partial_rotary_factor * head_dim // 2)``；
    - 频率指数按**完整** head_dim 归一（"proportional" 的含义，旋转部分的
      频率范围与全维 RoPE 成比例）：
      ``inv_freq = 1 / theta_base ** (arange(0, 2 * rope_angles, 2) / head_dim)``；
    - 未旋转的频率对 inv_freq 置 0，对应角度恒为 0，即 cos=1/sin=0 的
      恒等旋转，因此可直接复用现有 ``compute_rope``（split-halves 风格）。

    Returns:
        (cos, sin)，各 (context_length, head_dim)
    """
    assert head_dim % 2 == 0, "Embedding dimension must be even"

    rope_angles = int(partial_rotary_factor * head_dim // 2)
    inv_freq_rotated = 1.0 / (
        theta_base
        ** (torch.arange(0, 2 * rope_angles, 2, dtype=dtype) / head_dim)
    )
    nope_angles = head_dim // 2 - rope_angles
    if nope_angles > 0:
        inv_freq = torch.cat(
            (inv_freq_rotated, torch.zeros(nope_angles, dtype=dtype)), dim=0
        )
    else:
        inv_freq = inv_freq_rotated

    positions = torch.arange(context_length, dtype=dtype)
    angles = positions[:, None] * inv_freq[None, :]
    angles = torch.cat((angles, angles), dim=1)

    return angles.cos(), angles.sin()


def _with_head_dim(cfg, head_dim):
    """返回把 head_dim 覆盖为指定值的浅拷贝 cfg（相同则原样返回）。"""
    if head_dim == getattr(cfg, "head_dim", None):
        return cfg
    layer_cfg = copy.copy(cfg)
    layer_cfg.head_dim = head_dim
    return layer_cfg


class Model(nn.Module):
    """
    Gemma 4 text-only 教学模型。

    复用 ``TransformerBlockGemma3``（pre-norm + post-norm 的 Gemma 风格残差、
    GQA + q/k RMSNorm）。full_attention 层通过逐层 cfg 覆盖为
    ``global_head_dim``，以复刻官方"全局层加宽 head_dim"的行为。
    """

    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        # layer_types：优先取显式配置（同 Gemma 3），否则按交错比生成
        # （DotDict 缺键抛 KeyError，SimpleNamespace/ModelConfig 缺属性抛
        # AttributeError，二者都视为未提供）
        try:
            layer_types = cfg.layer_types or None
        except (AttributeError, KeyError):
            layer_types = None
        layer_types = layer_types or layer_pattern(
            cfg.n_layers, getattr(cfg, "local_per_global", 5)
        )
        assert len(layer_types) == cfg.n_layers
        self.layer_types = layer_types

        global_head_dim = getattr(cfg, "global_head_dim", None) or cfg.head_dim

        # main model parameters
        self.tok_embed = nn.Embedding(cfg.vocab_size, cfg.embed_dim, dtype=cfg.dtype)
        self.blocks = nn.ModuleList([
            TransformerBlockGemma3(
                _with_head_dim(
                    cfg,
                    global_head_dim if attn_type == "full_attention" else cfg.head_dim,
                ),
                attn_type,
            )
            for attn_type in layer_types
        ])
        self.final_norm = RMSNorm_Gemma3(cfg.embed_dim, eps=1e-6)
        self.out_head = nn.Linear(cfg.embed_dim, cfg.vocab_size, bias=False, dtype=cfg.dtype)
        # 权重共享：官方 Gemma 4 tie_word_embeddings=True
        self.out_head.weight = self.tok_embed.weight
        # reusable utilities
        cos_local, sin_local = precompute_rope_params(
            head_dim=cfg.head_dim,
            theta_base=cfg.rope_local_base,
            context_length=cfg.context_length,
            dtype=torch.float32,
        )
        cos_global, sin_global = precompute_proportional_rope_params(
            head_dim=global_head_dim,
            theta_base=cfg.rope_base,
            partial_rotary_factor=getattr(cfg, "partial_rotary_factor", 1.0),
            context_length=cfg.context_length,
            dtype=torch.float32,
        )
        self.register_buffer("cos_local", cos_local, persistent=False)
        self.register_buffer("sin_local", sin_local, persistent=False)
        self.register_buffer("cos_global", cos_global, persistent=False)
        self.register_buffer("sin_global", sin_global, persistent=False)

    def _create_masks(self, seq_len, device):
        ones = torch.ones((seq_len, seq_len), dtype=torch.bool, device=device)

        # mask_global (future is masked: j > i)
        mask_global = torch.triu(ones, diagonal=1)

        # far_past (too far back is masked: i - j >= sliding_window)
        far_past = torch.triu(ones, diagonal=self.cfg.sliding_window).T

        # Local (sliding_window) = future OR far-past
        mask_local = mask_global | far_past

        return mask_global, mask_local

    def forward(self, x):
        # Forward pass
        batch_size, seq_len = x.shape
        x = self.tok_embed(x) * (self.cfg.embed_dim ** 0.5)
        mask_global, mask_local = self._create_masks(seq_len, x.device)
        for block in self.blocks:
            x = block(
                x,
                mask_global=mask_global,
                mask_local=mask_local,
                cos_global=self.cos_global,
                sin_global=self.sin_global,
                cos_local=self.cos_local,
                sin_local=self.sin_local,
            )
        x = self.final_norm(x)
        logits = self.out_head(x.to(self.cfg.dtype))
        # final_logit_softcapping（官方 30.0）：tanh 软限幅
        softcap = getattr(self.cfg, "final_logit_softcapping", None)
        if softcap:
            logits = torch.tanh(logits / softcap) * softcap

        return logits




# 测试代码 main 函数
def main():
    from utils.log_util import logger
    from utils.args_tools import DotDict
    from config.gemma4.model_cfgs import GEMMA4_TINY_CONFIG

    # ------------------------------
    # parms
    # ------------------------------
    cfg = DotDict(GEMMA4_TINY_CONFIG)
    # ------------------------------
    # random seed
    # ------------------------------
    torch.manual_seed(123)
    # ------------------------------
    # model
    # ------------------------------
    model = Model(cfg)
    logger.info(model)
    logger.info(f"layer_types: {model.layer_types}")
    # ------------------------------
    # model test
    # ------------------------------
    model_output = model(torch.tensor([1, 2, 3]).unsqueeze(0))
    logger.info(f"model_output.shape: {model_output.shape}")
    # ------------------------------
    # model memory
    # ------------------------------
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total number of parameters: {total_params:,}")
    total_params_normalized = total_params - model.tok_embed.weight.numel()
    logger.info(f"Total number of unique parameters: {total_params_normalized:,}")


if __name__ == "__main__":
    main()
