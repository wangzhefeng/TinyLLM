# -*- coding: utf-8 -*-

# ***************************************************
# * File        : mtp.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2026-10-24
# * Version     : 1.0.102424
# * Description : DeepSeek-V3 MTP（Multi-Token Prediction，多 token 预测）教学实现。
# *               MTP 是 V3 训练专用组件：第 k 个 MTP 模块把主模型（或上一级
# *               MTP）第 i 个位置的隐状态 h_prev 与第 i+k 个 token 的 embedding
# *               拼接，经 RMSNorm 归一化、线性投影降维后送入一个 transformer
# *               block，再用与主模型共享的 norm/out_head 预测第 i+k+1 个 token。
# *               多个 MTPHead 可级联（h_new 传给下一级），对应论文 D 个顺序
# *               模块预测 D 个额外 token；embedding / final_norm / out_head 与
# *               主模型共享，通过 DeepSeekModel.attach_mtp() 挂载。
# * Link        : DeepSeek-V3 https://arxiv.org/abs/2412.19437 §2.5.1
# * Requirement : torch
# ***************************************************

__all__ = [
    "MTPHead",
    "deepseek_mtp_loss",
    "deepseek_mtp_joint_loss",
    "deepseek_mtp_train_step",
]

# python libraries
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.normalization.rms_norm import RMSNorm
from models.deepseek.model import DeepSeekBlock

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


class MTPHead(nn.Module):
    """DeepSeek-V3 §2.5.1 的单个 MTP 模块。

    结构（对应论文图 3 的一个 MTP Module）：

    1. ``concat([h_prev, embed(x)])``：上一级隐状态与目标 token embedding 拼接；
    2. ``combine_norm``：RMSNorm 归一化拼接结果（``2 * embed_dim`` 维）；
    3. ``projection``：线性投影降回 ``embed_dim``（论文中的投影矩阵 M）；
    4. ``trf_block``：一个复用 ``DeepSeekBlock`` 的 transformer block；
    5. 共享的 ``final_norm`` + ``out_head`` 产生 logits；
       同时输出 ``h_new`` 供下一深度 MTP 级联。

    Args:
        cfg: 与 ``DeepSeekModel`` 相同的配置
        shared_embedding: 共享的 token embedding（通常为主模型 ``tok_embed``）
        shared_norm: 共享的输出 RMSNorm；None 时自建
        shared_out_head: 共享的输出投影；None 时自建
    """

    def __init__(self, cfg, shared_embedding, shared_norm=None, shared_out_head=None):
        super().__init__()

        self.shared_embedding = shared_embedding
        self.combine_norm = RMSNorm(2 * cfg.embed_dim, eps=1e-6)
        self.projection = nn.Linear(2 * cfg.embed_dim, cfg.embed_dim, bias=False, dtype=cfg.dtype)
        self.trf_block = DeepSeekBlock(cfg)
        self.shared_norm = shared_norm if shared_norm is not None else RMSNorm(cfg.embed_dim, eps=1e-6)
        self.shared_out_head = (
            shared_out_head if shared_out_head is not None
            else nn.Linear(cfg.embed_dim, cfg.vocab_size, bias=False, dtype=cfg.dtype)
        )

    def forward(self, input_ids, h_prev, start_pos: int = 0, cache=None):
        """前向。

        Args:
            input_ids: (B, T)，目标 token（相对主模型输入右移 k 位）
            h_prev: (B, T, embed_dim)，主模型或上一级 MTP 的隐状态
            start_pos: 增量解码起始位置，语义与 ``DeepSeekBlock`` 一致
            cache: ``trf_block`` 内 MLA 的 ``(c_kv, k_rope)`` 元组；None 表示全新序列

        Returns:
            (logits, h_new)：logits 形状 (B, T, vocab_size)，
            h_new 形状 (B, T, embed_dim)，供下一深度 MTP 级联。
        """
        x = torch.cat([h_prev, self.shared_embedding(input_ids)], dim=-1)
        x = self.combine_norm(x)
        x = self.projection(x)
        h_new, _ = self.trf_block(x, start_pos=start_pos, cache=cache)
        logits = self.shared_out_head(self.shared_norm(h_new).to(self.shared_out_head.weight.dtype))
        return logits, h_new


def deepseek_mtp_loss(logits, target_ids):
    """MTP 训练目标：逐位置交叉熵（论文式 2.5.1 的 L_MTP 单项）。

    Args:
        logits: (B, T, vocab_size)，MTPHead 输出
        target_ids: (B, T)，右移后的目标 token id

    Returns:
        标量交叉熵损失。
    """
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]).float(),
        target_ids.reshape(-1),
    )


def deepseek_mtp_joint_loss(model, input_ids, mtp_weight: float = 1.0,
                            return_parts: bool = False):
    """MTP 联合训练损失（论文式 2.5.1 的 L_total）。

    主模型 CE + mtp_weight · mean_d(L_MTP_d)：第 d 个 MTP 头以主模型（级联时
    上一级）位置 t 的隐状态与 token_{t+d+1} 的 embedding 为输入，预测
    token_{t+d+2}；各深度对齐到 t = 0..T-d-3。

    Args:
        model: 已调用 ``attach_mtp`` 的 ``DeepSeekModel``
        input_ids: (B, T)，T 需 >= n_mtp + 3
        mtp_weight: MTP 项权重
        return_parts: True 时返回 (total, main_loss, per_depth_losses)

    Returns:
        total 或 (total, main_loss, mtp_losses)
    """
    if getattr(model, "mtp_heads", None) is None:
        raise RuntimeError("先调用 model.attach_mtp(n_mtp) 再计算联合损失")
    batch_size, seq_len = input_ids.shape
    n_depth = len(model.mtp_heads)
    if seq_len < n_depth + 3:
        raise ValueError(
            f"序列长度 {seq_len} 不足以训练 {n_depth} 个 MTP 深度（需 >= {n_depth + 3}）")
    logits, hidden = model(input_ids, output_hidden_states=True)
    main_loss = F.cross_entropy(
        logits[:, :-1].reshape(-1, logits.shape[-1]).float(),
        input_ids[:, 1:].reshape(-1),
    )
    h_prev = hidden
    mtp_losses = []
    for d, head in enumerate(model.mtp_heads):
        length = seq_len - d - 2
        tokens_d = input_ids[:, 1 + d : 1 + d + length]
        targets_d = input_ids[:, 2 + d : 2 + d + length]
        logits_d, h_new = head(tokens_d, h_prev[:, :length])
        mtp_losses.append(deepseek_mtp_loss(logits_d, targets_d))
        h_prev = h_new
    mtp_loss = torch.stack(mtp_losses).mean()
    total = main_loss + mtp_weight * mtp_loss
    if return_parts:
        return total, main_loss, torch.stack(mtp_losses)
    return total


def deepseek_mtp_train_step(model, optimizer, input_ids, mtp_weight: float = 1.0):
    """MTP 联合训练的单步封装（前向 -> 联合损失 -> 反传 -> 更新）。

    Returns:
        (total, main_loss, mtp_losses) 的 detach 标量
    """
    total, main_loss, mtp_losses = deepseek_mtp_joint_loss(
        model, input_ids, mtp_weight=mtp_weight, return_parts=True)
    optimizer.zero_grad()
    total.backward()
    optimizer.step()
    return (
        float(total.item()),
        float(main_loss.item()),
        mtp_losses.detach(),
    )


# 测试代码 main 函数
def main():
    import torch

    from utils.args_tools import DotDict
    from utils.log_util import logger
    from models.deepseek.model import DeepSeekModel

    # DeepSeek-tiny 教学配置（与 models/deepseek/model.py main 一致）
    DEEPSEEK_TINY_CONFIG = {
        "vocab_size": 1000,
        "context_length": 256,
        "embed_dim": 256,
        "n_layers": 4,
        "n_heads": 8,
        "kv_lora_rank": 64,           # V3: 512
        "q_lora_rank": 128,           # V3: 1536
        "qk_nope_head_dim": 24,       # V3: 128
        "qk_rope_head_dim": 8,        # V3: 64
        "hidden_dim": 512,            # FFN 中间维（FeedForwardSiLU 回退键）
        "rope_base": 10_000.0,
        "dtype": torch.bfloat16,
    }
    cfg = DotDict(DEEPSEEK_TINY_CONFIG)

    torch.manual_seed(123)
    model = DeepSeekModel(cfg).eval()
    model.attach_mtp(1)
    head = model.mtp_heads[0]

    x = torch.randint(0, cfg["vocab_size"], (2, 8))
    h_prev = torch.randn(2, 8, cfg["embed_dim"], dtype=cfg["dtype"])
    with torch.no_grad():
        logits, h_new = head(x, h_prev)
    logger.info(f"MTPHead forward: logits {tuple(logits.shape)}, h_new {tuple(h_new.shape)}")

    target = torch.randint(0, cfg["vocab_size"], (2, 8))
    loss = deepseek_mtp_loss(logits, target)
    logger.info(f"deepseek_mtp_loss: {loss.item():.4f}")

    # 共享参数只计一次：模型总参数 = 主模型 + combine_norm + projection + trf_block
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"params (with 1 MTP head, shared counted once): {total:,}")

if __name__ == "__main__":
    main()
