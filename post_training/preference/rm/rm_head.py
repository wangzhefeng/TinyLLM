# -*- coding: utf-8 -*-

# ***************************************************
# * File        : rm_head.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2026-10-24
# * Version     : 1.0.102421
# * Description : Reward Model（RLHF 一阶段，from scratch）：
# *               causal LM backbone + 标量价值头；序列得分取最后一个非
# *               pad token 位置的 reward（右 padding 约定，与
# *               post_training/preference/data_provider/dpo 的 collate 一致）；
# *               成对偏好用 Bradley-Terry 损失 -log σ(r_chosen - r_rejected)。
# *               训练编排见 exp/exp_train_rm.py，数据复用 DPO 的偏好对。
# * Link        : InstructGPT https://arxiv.org/abs/2203.02155
# * Requirement : torch
# ***************************************************

__all__ = ["RewardModel", "bradley_terry_loss", "rm_pairwise_step"]

# python libraries
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


class RewardModel(nn.Module):
    """序列 reward 模型：backbone 输出逐 token 标量 reward。

    Args:
        backbone: causal LM（如 models.gpt2.Model，forward 需支持
            ``output_hidden_states=True`` 返回 (logits, final_hidden)）
        cfg: 模型配置（取 embed_dim；缺省时从 backbone.tok_embed 推断）
    """

    def __init__(self, backbone, cfg=None):
        super().__init__()

        self.backbone = backbone
        embed_dim = getattr(cfg, "embed_dim", None)
        if embed_dim is None:
            embed_dim = backbone.tok_embed.embedding_dim
        self.v_head = nn.Linear(embed_dim, 1)

    def token_rewards(self, input_ids, attention_mask=None, position_ids=None):
        """逐 token reward，shape (B, T)。"""
        _, hidden = self.backbone(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=True,
        )
        return self.v_head(hidden).squeeze(-1)

    def forward(self, input_ids, pad_token_id=50256, attention_mask=None, position_ids=None):
        """序列得分：最后一个非 pad token 位置的 reward，shape (B,)。"""
        rewards = self.token_rewards(input_ids, attention_mask, position_ids)
        lengths = (input_ids != pad_token_id).sum(dim=1).clamp(min=1)
        idx = (lengths - 1).view(-1, 1)
        return rewards.gather(1, idx).squeeze(-1)


def bradley_terry_loss(r_chosen, r_rejected):
    """Bradley-Terry 成对偏好损失：``-log sigmoid(r_chosen - r_rejected)``。"""
    return -F.logsigmoid(r_chosen - r_rejected).mean()


def rm_pairwise_step(model, optimizer, chosen_ids, rejected_ids, pad_token_id=50256):
    """单个成对 batch 的 RM 训练步（前向 -> BT loss -> 反传 -> 更新）。

    Returns:
        (loss, reward_margin)：标量损失与 margin = mean(r_chosen - r_rejected)，
        margin > 0 表示模型当前偏好 chosen 一侧
    """
    r_chosen = model(chosen_ids, pad_token_id=pad_token_id)
    r_rejected = model(rejected_ids, pad_token_id=pad_token_id)
    loss = bradley_terry_loss(r_chosen, r_rejected)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    with torch.no_grad():
        margin = (r_chosen - r_rejected).mean()
    return float(loss.item()), float(margin.item())


# 测试代码 main 函数（规范测试见 tests/unit/test_rm.py）
def main():
    from types import SimpleNamespace

    import torch

    from models.gpt2.gpt2_124M import Model

    torch.manual_seed(123)
    cfg = SimpleNamespace(
        vocab_size=1000, context_length=64, embed_dim=64, d_ff=128,
        n_heads=4, n_layers=2, dropout=0.0, qkv_bias=True, dtype=torch.float32,
    )
    rm = RewardModel(Model(cfg), cfg)
    ids = torch.randint(0, 1000, (2, 16))
    optimizer = torch.optim.AdamW(rm.parameters(), lr=1e-3)
    loss, margin = rm_pairwise_step(rm, optimizer, ids, torch.flip(ids, [1]))
    print(f"loss={loss:.4f}, margin={margin:.4f}")

if __name__ == "__main__":
    main()
