# -*- coding: utf-8 -*-

# ***************************************************
# * File        : reinforce.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2026-10-21
# * Version     : 1.0.102121
# * Description : REINFORCE / REINFORCE-with-baseline from scratch
# *               （reasoning-from-scratch ch06 的渐进教学链第一环：
# *               先于 GRPO 理解"策略梯度 = reward 加权的 log-prob 梯度"）。
# *               1. REINFORCE：∇J = E[r · ∇log π(τ)]
# *               2. REINFORCE-with-baseline：∇J = E[(r - b) · ∇log π(τ)]
# *                  b 不引入偏差、只降方差（组均值/批均值 baseline）
# *               3. GRPO（见 reinforcement_learning.py）进一步把组内
# *                  (r - mean)/std 归一化为优势并加 PPO 式 clip
# * Link        : REINFORCE https://link.springer.com/article/10.1007/BF00992696
# *               reasoning-from-scratch ch06
# * Requirement : torch
# ***************************************************

__all__ = [
    "compute_reinforce_loss",
    "reinforce_step",
]

# python libraries
from pathlib import Path

import torch

from post_training.rl import generate_group_samples, token_logprobs

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


def compute_reinforce_loss(seq_logprobs, rewards, baseline=None):
    """REINFORCE(-with-baseline) 策略梯度损失（负号用于最小化）。

    ``L = -mean_i( (r_i - b) · sum_t log π(a_t|s_t) )``

    Args:
        seq_logprobs: (N,) 每条生成序列的 log-prob（序列级求和）
        rewards: (N,) 每条生成的 reward
        baseline: None（纯 REINFORCE）/ "mean"（批均值 baseline，
            GRPO 组均值思想的退化形式）/ 标量张量（外部价值估计）

    Returns:
        标量 loss（可反传到 seq_logprobs 的来源模型）
    """
    if baseline is None:
        adv = rewards
    elif baseline == "mean":
        adv = rewards - rewards.mean()
    else:
        adv = rewards - baseline
    return -(adv * seq_logprobs).mean()


def reinforce_step(model, optimizer, prompt_ids, reward_fn,
                   num_samples: int = 4, max_new_tokens: int = 32,
                   context_length=None, temperature=1.0, top_k=None, top_p=None,
                   eos_token_id=None, baseline="mean", decode_fn=None,
                   max_grad_norm: float = 1.0):
    """单步 REINFORCE(-with-baseline) 更新（reasoning ch06 教学链）。

    对同一 prompt 采样 ``num_samples`` 条生成（与 GRPO 的 group 同构，
    但不做组内 std 归一化、不做重要性采样比率裁剪——突出裸策略梯度
    与 baseline 降方差的效果，作为 GRPO 的对照组）。

    Args:
        reward_fn: callable[token_ids_row 或 str] -> float
        baseline: None / "mean" / 标量；默认 "mean"（with-baseline 版）
        max_grad_norm: 梯度裁剪；None 关闭

    Returns:
        dict(loss=..., mean_reward=..., rewards=..., baseline=...)
    """
    # 复用 GRPO 的组采样（返回 samples / old_logprobs / completion_lengths；
    # REINFORCE 是 on-policy 单轮更新，old_logprobs 即当前策略 logprob）
    samples, old_logprobs, completion_lengths = generate_group_samples(
        model, prompt_ids, group_size=num_samples, max_new_tokens=max_new_tokens,
        context_length=context_length, temperature=temperature,
        top_k=top_k, top_p=top_p, eos_token_id=eos_token_id,
    )
    # 序列级 logprob：只累加真实生成 token（排除 padding）
    mask = torch.arange(old_logprobs.shape[1], device=samples.device).unsqueeze(0) \
        < completion_lengths.unsqueeze(1)
    seq_logprobs = (old_logprobs * mask).sum(dim=1)

    rewards = []
    for row in samples:
        arg = decode_fn(row) if decode_fn is not None else row
        rewards.append(float(reward_fn(arg)))
    rewards = torch.tensor(rewards, device=samples.device)

    # 重新以带梯度方式前向，得到可反传的序列 logprob
    grad_logprobs = token_logprobs(model, samples, start_pos=prompt_ids.shape[1])
    grad_seq_logprobs = (grad_logprobs * mask).sum(dim=1)

    if baseline is None:
        b = None
    elif baseline == "mean":
        b = rewards.mean()
    else:
        b = torch.as_tensor(baseline, device=samples.device)
    loss = compute_reinforce_loss(grad_seq_logprobs, rewards, baseline=b)

    optimizer.zero_grad()
    loss.backward()
    if max_grad_norm is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()

    return {
        "loss": float(loss.item()),
        "mean_reward": float(rewards.mean().item()),
        "rewards": rewards,
        "baseline": None if b is None else float(b.item()),
    }


# 测试代码 main 函数
def main():
    # 用法示例见 tests/unit/test_reinforce.py
    pass

if __name__ == "__main__":
    main()
