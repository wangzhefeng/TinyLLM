# -*- coding: utf-8 -*-

# ***************************************************
# * File        : main.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2026-10-21
# * Version     : 1.0.102121
# * Description : PPO（Proximal Policy Optimization）from scratch，
# *               面向 LLM 的**序列级 bandit**教学实现（reasoning-from-scratch
# *               ch06 教学链的又一环，与 reasoning/training/reinforcement_learning.py
# *               的 GRPO 形成对照）：
# *               1. 对同一 prompt 采样 N 条生成，缓存旧策略 logprob
# *               2. critic（价值头）估计 V(s)，优势 A_i = r_i - V(s)
# *                  （不做组内归一化——这正是与 GRPO 的核心区别：
# *                  GRPO 用组均值/std 替代 critic；PPO 用学习到的价值函数）
# *               3. token 级重要性采样比率 + PPO clip 目标
# *                  + 可选 k3 KL 惩罚 + completion mask 排除 padding
# *               4. critic 损失 MSE(V(s), r)，与 actor 分开反传/分开更新
# * Link        : PPO https://arxiv.org/abs/1707.06347
# *               reasoning-from-scratch ch06
# * Requirement : torch
# ***************************************************

__all__ = [
    "CriticValueHead",
    "compute_advantages_ppo",
    "compute_ppo_actor_loss",
    "compute_critic_loss",
    "ppo_update_step",
]

# python libraries
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from post_training.reasoning.training.reinforcement_learning import (
    completion_mask,
    generate_group_samples,
    token_logprobs,
)

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


class CriticValueHead(nn.Module):
    """价值头：prompt 末 token 的 token embedding -> 标量 V(s)。

    教学简化：bandit 设定下"状态"= prompt，critic 直接对 prompt 末 token 的
    token embedding 做线性映射（不上变压器主干，避免价值损失污染表征）。
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.linear = nn.Linear(embed_dim, 1)

    def forward(self, embeds: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeds: (B, embed_dim)（prompt 末 token 的 embedding）

        Returns:
            (B,) 价值估计 V(s)
        """
        return self.linear(embeds).squeeze(-1)


def compute_advantages_ppo(rewards: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    """PPO 优势：``A_i = r_i - V(s)``（不做组内归一化，与 GRPO 的对照点）。

    Args:
        rewards: (N,) 每条生成的序列 reward
        values: (N,) critic 对共享 prompt 的价值估计（调用方负责 detach）

    Returns:
        (N,) 优势
    """
    return rewards - values


def compute_ppo_actor_loss(model, samples, old_logprobs, advantages, prompt_len,
                           clip_eps: float = 0.2, kl_beta: float = 0.0,
                           ref_model=None, completion_lengths=None):
    """PPO clipped 策略目标（token 级比率裁剪 + 可选 KL 惩罚）。

    ``L = -masked_mean( min(ratio·A, clip(ratio, 1±eps)·A) ) + kl_beta · masked_mean(KL)``

    与 compute_grpo_loss 结构相同（同一 clipped surrogate），
    区别在于优势的来源：这里 A = r - V(s)（critic），GRPO 是组内归一化。

    Args:
        model: 当前策略（可训练）
        samples: (N, T+L) 生成序列
        old_logprobs: (N, L) 采样时的旧策略 log-prob
        advantages: (N,) 优势（broadcast 到每个 token）
        prompt_len: prompt 长度（只对生成部分计损）
        clip_eps: 比率裁剪半径
        kl_beta: KL 惩罚系数（0 = 不加）；需配合 ref_model
        ref_model: 冻结的参考策略（通常 = 初始模型）
        completion_lengths: (N,) 每条生成的真实 completion 长度；
            提供时 padding 位不进入梯度与 KL
    """
    new_logprobs = token_logprobs(model, samples, start_pos=prompt_len)  # (N, L)
    ratio = torch.exp(new_logprobs - old_logprobs)                       # (N, L)
    adv = advantages.unsqueeze(1)                                        # (N, 1)
    unclipped = ratio * adv
    clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv
    token_obj = torch.min(unclipped, clipped)                            # (N, L)

    if completion_lengths is not None:
        mask = completion_mask(completion_lengths, new_logprobs.shape[1]).float()
        policy_loss = -(token_obj * mask).sum() / mask.sum().clamp(min=1.0)
    else:
        policy_loss = -token_obj.mean()

    if kl_beta > 0.0 and ref_model is not None:
        with torch.no_grad():
            ref_logprobs = token_logprobs(ref_model, samples, start_pos=prompt_len)
        # 逐 token 前向 KL 近似：log(π/π_ref) → exp(x) - x - 1（k3 估计，非负）
        log_ratio = new_logprobs - ref_logprobs
        kl = torch.exp(log_ratio) - log_ratio - 1.0
        if completion_lengths is not None:
            kl_term = (kl * mask).sum() / mask.sum().clamp(min=1.0)
        else:
            kl_term = kl.mean()
        policy_loss = policy_loss + kl_beta * kl_term
    return policy_loss


def compute_critic_loss(values: torch.Tensor, rewards: torch.Tensor) -> torch.Tensor:
    """critic 价值损失：MSE(V(s), r)（bandit 设定下每条生成共享同一状态 prompt）。

    Args:
        values: (N,) critic 价值估计（带梯度）
        rewards: (N,) 环境 reward（监督目标）

    Returns:
        标量 MSE
    """
    return F.mse_loss(values, rewards)


def ppo_update_step(model, optimizer, prompt_ids, reward_fn,
                    num_samples: int = 4, max_new_tokens: int = 32,
                    context_length=None, temperature=1.0, top_k=None, top_p=None,
                    eos_token_id=None, clip_eps: float = 0.2, kl_beta: float = 0.0,
                    ref_model=None, decode_fn=None, max_grad_norm: float = 1.0,
                    critic=None, critic_optimizer=None, value_fn=None):
    """单步 PPO 更新（序列级 bandit 版，reasoning ch06 教学链）。

    流程：采样 N 条 -> reward_fn 打分 -> critic 估计 V(s) -> 优势 A = r - V
          -> actor clipped PG loss（反传 model）-> critic MSE loss（反传 critic）。
    actor 与 critic **分开反传、分开 step**（各自 zero_grad/clip/step）：
    教学上强调两个网络的梯度不应互相污染——critic 输入已用 detach 切断
    对 token embedding 的梯度，价值损失不会流回 model。

    Args:
        reward_fn: callable[token_ids_row 或 str] -> float
        critic: CriticValueHead（None 时退化：value=0，即纯 REINFORCE）
        critic_optimizer: critic 的优化器；critic 不为 None 时必填
        value_fn: callable[prompt_ids] -> (N,) 张量；注入式 baseline
            （测试可传常数），与 critic 互斥
        max_grad_norm: actor 与 critic 各自的梯度范数裁剪；None 关闭

    Returns:
        dict(loss=actor+critic 合计, actor_loss, critic_loss, mean_reward,
             advantages, rewards, values)
    """
    if critic is not None and value_fn is not None:
        raise ValueError("critic 与 value_fn 互斥：二者只能提供一个")
    if critic is not None and critic_optimizer is None:
        raise ValueError("提供 critic 时必须提供 critic_optimizer（actor/critic 分开更新）")

    # 1) 采样 N 条生成（缓存旧策略 logprob 与真实 completion 长度）
    samples, old_logprobs, completion_lengths = generate_group_samples(
        model, prompt_ids, group_size=num_samples, max_new_tokens=max_new_tokens,
        context_length=context_length, temperature=temperature,
        top_k=top_k, top_p=top_p, eos_token_id=eos_token_id,
    )
    # 2) reward 打分
    rewards = []
    for row in samples:
        arg = decode_fn(row) if decode_fn is not None else row
        rewards.append(float(reward_fn(arg)))
    rewards = torch.tensor(rewards, device=samples.device)

    # 3) 价值估计：N 条生成共享同一 prompt 状态（bandit）
    if value_fn is not None:
        values = torch.as_tensor(value_fn(prompt_ids), dtype=torch.float32,
                                 device=samples.device).detach()
        critic_loss = torch.tensor(0.0, device=samples.device)
    elif critic is not None:
        # critic 前向带梯度（供 MSE 反传）；detach embedding 切断对 model 的梯度
        prompt_last_embed = model.tok_embed(prompt_ids[:, -1]).detach()  # (1, embed_dim)
        values_with_grad = critic(prompt_last_embed.expand(num_samples, -1))  # (N,)
        critic_loss = compute_critic_loss(values_with_grad, rewards)
        values = values_with_grad.detach()
    else:
        values = torch.zeros(num_samples, device=samples.device)
        critic_loss = torch.tensor(0.0, device=samples.device)

    # 4) 优势：r - V（不组归一化）
    advantages = compute_advantages_ppo(rewards, values)

    # 5) actor 更新（只动 model）
    actor_loss = compute_ppo_actor_loss(
        model, samples, old_logprobs, advantages,
        prompt_len=prompt_ids.shape[1], clip_eps=clip_eps,
        kl_beta=kl_beta, ref_model=ref_model,
        completion_lengths=completion_lengths,
    )
    optimizer.zero_grad()
    actor_loss.backward()
    if max_grad_norm is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()

    # 6) critic 更新（只动 critic，分开反传/分开 step）
    if critic is not None:
        critic_optimizer.zero_grad()
        critic_loss.backward()
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(critic.parameters(), max_grad_norm)
        critic_optimizer.step()

    return {
        "loss": float(actor_loss.item()) + float(critic_loss.item()),
        "actor_loss": float(actor_loss.item()),
        "critic_loss": float(critic_loss.item()),
        "mean_reward": float(rewards.mean().item()),
        "advantages": advantages,
        "rewards": rewards,
        "values": values,
    }


# 测试代码 main 函数
def main():
    # 用法示例见 tests/unit/test_ppo.py
    pass

if __name__ == "__main__":
    main()
