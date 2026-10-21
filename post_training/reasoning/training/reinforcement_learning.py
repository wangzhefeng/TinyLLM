# -*- coding: utf-8 -*-

# ***************************************************
# * File        : reinforcement_learning.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2026-10-20
# * Version     : 2.0.102021
# * Description : GRPO（Group Relative Policy Optimization）from scratch，
# *               对应 reasoning-from-scratch ch06/ch07 的 RLVR 主线。
# *               纯 PyTorch 教学实现（不依赖 unsloth/vllm/trl）：
# *               1. 对每个 prompt 采样 G 条生成（group）
# *               2. reward_fn 打分，组内相对优势 A = (r - mean) / std
# *               3. PPO 式 clipped policy gradient + 可选 KL 惩罚
# *               4. 旧策略（old policy）log-prob 在采样时缓存，作为比率分母
# * Link        : DeepSeekMath/GRPO https://arxiv.org/abs/2402.03300
# *               reasoning-from-scratch ch06/ch07
# * Requirement : torch
# ***************************************************

__all__ = [
    "compute_group_advantages",
    "compute_grpo_loss",
    "completion_mask",
    "grpo_update_step",
    "grpo_update_epoch",
    "generate_group_samples",
    "token_logprobs",
]

# python libraries
from pathlib import Path

import torch
import torch.nn.functional as F

from inference.generate_core import sample_next_token

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


def _token_logprobs(model, token_ids, start_pos=0):
    """计算自回归模型对 ``token_ids[:, start_pos:]`` 每个 token 的 log-prob。

    Args:
        model: forward(x) 返回 (B, T, V) logits
        token_ids: (B, T) 完整序列（prompt + 生成）
        start_pos: 从该位置起计入 log-prob（即 prompt 长度）

    Returns:
        (B, T - start_pos) 每个生成 token 的 log-prob
    """
    logits = model(token_ids)[:, start_pos - 1:-1, :]      # (B, L, V)，预测位 i+1
    logprobs = F.log_softmax(logits, dim=-1)
    targets = token_ids[:, start_pos:]                      # (B, L)
    return torch.gather(logprobs, -1, targets.unsqueeze(-1)).squeeze(-1)


# 公开别名：供 reinforce.py 等同族实现复用（reasoning ch06 的 REINFORCE 家族）
token_logprobs = _token_logprobs


def completion_mask(completion_lengths, max_len: int) -> torch.Tensor:
    """由每条生成的真实 completion 长度构造 (G, max_len) 掩码。

    padding 位（提前 EOS 后用末 token 补齐的位置）为 False，
    真实生成位为 True；供 loss/KL 只统计有效 token。
    """
    pos = torch.arange(max_len, device=completion_lengths.device).unsqueeze(0)
    return pos < completion_lengths.unsqueeze(1)


@torch.no_grad()   # 不能用 inference_mode：其产出的张量无法参与后续 backward
def generate_group_samples(model, prompt_ids, group_size=4, max_new_tokens=32,
                           context_length=None, temperature=1.0, top_k=None,
                           top_p=None, eos_token_id=None):
    """对同一 prompt 采样 ``group_size`` 条生成（GRPO 的 group）。

    同时缓存**旧策略**（当前模型）的 log-prob，作为后续重要性采样比率的
    分母（``ratio = exp(logp_new - logp_old)``）。

    Returns:
        samples: (group_size, T + L) token ids（组内最长者为准，右侧补齐）
        old_logprobs: (group_size, L) 旧策略 log-prob
        completion_lengths: (group_size,) 每条生成的真实 completion 长度
            （≤ L；提前 EOS 的序列小于 L，padding 位由 completion_mask 排除）
    """
    model.eval()
    prompt_len = prompt_ids.shape[1]
    ctx_len = context_length or prompt_len + max_new_tokens
    samples = []
    comp_lens = []
    for _ in range(group_size):
        seq = prompt_ids
        generated = 0
        for _ in range(max_new_tokens):
            logits = model(seq[:, -ctx_len:])
            next_idx = sample_next_token(logits[:, -1, :], temperature=temperature,
                                         top_k=top_k, top_p=top_p)
            seq = torch.cat([seq, next_idx], dim=1)
            generated += 1
            if eos_token_id is not None and bool((next_idx == eos_token_id).all()):
                break
        samples.append(seq)
        comp_lens.append(generated)
    # 补齐到等长（pad 用最后一个 token 即可——教学实现，注意力 mask 省略）
    max_len = max(s.shape[1] for s in samples)
    padded = []
    for s in samples:
        pad_len = max_len - s.shape[1]
        padded.append(torch.cat([s, s[:, -1:].expand(1, pad_len)], dim=1))
    samples = torch.cat(padded, dim=0)                     # (G, T+L)
    completion_lengths = torch.tensor(comp_lens, device=samples.device)
    # 旧策略 log-prob（无梯度）
    old_logprobs = _token_logprobs(model, samples, start_pos=prompt_len)
    # clone：inference_mode 产出的张量不能进 autograd（后续 backward 会报错）
    return samples.clone(), old_logprobs.clone(), completion_lengths


def compute_group_advantages(rewards: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """GRPO 组内相对优势：``A_i = (r_i - mean(group)) / (std(group) + eps)``。

    Args:
        rewards: (G,) 组内每条生成的 reward
    """
    mean = rewards.mean()
    std = rewards.std(unbiased=False)
    return (rewards - mean) / (std + eps)


def compute_grpo_loss(model, samples, old_logprobs, advantages, prompt_len,
                      clip_eps: float = 0.2, kl_beta: float = 0.0,
                      ref_model=None, completion_lengths=None):
    """GRPO clipped loss（PPO 式比率裁剪 + 可选 KL 惩罚）。

    ``L = -masked_mean( min(ratio·A, clip(ratio, 1±eps)·A) ) + kl_beta · masked_mean(KL)``

    Args:
        model: 当前策略（可训练）
        samples: (G, T+L) 生成序列
        old_logprobs: (G, L) 采样时的旧策略 log-prob
        advantages: (G,) 组内优势（broadcast 到每个 token）
        prompt_len: prompt 长度（只对生成部分计算 loss）
        clip_eps: 比率裁剪半径
        kl_beta: KL 惩罚系数（0 = 不加）；需配合 ref_model
        ref_model: 冻结的参考策略（通常 = 初始模型）
        completion_lengths: (G,) 每条生成的真实 completion 长度；
            提供时只对真实生成 token 计损（padding 位不进入梯度与 KL），
            None 时退化为全位置参与（旧教学行为）
    """
    new_logprobs = _token_logprobs(model, samples, start_pos=prompt_len)  # (G, L)
    ratio = torch.exp(new_logprobs - old_logprobs)                        # (G, L)
    adv = advantages.unsqueeze(1)                                          # (G, 1)
    unclipped = ratio * adv
    clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv
    token_obj = torch.min(unclipped, clipped)                             # (G, L)

    if completion_lengths is not None:
        mask = completion_mask(completion_lengths, new_logprobs.shape[1]).float()
        policy_loss = -(token_obj * mask).sum() / mask.sum().clamp(min=1.0)
    else:
        policy_loss = -token_obj.mean()

    if kl_beta > 0.0 and ref_model is not None:
        with torch.no_grad():
            ref_logprobs = _token_logprobs(ref_model, samples, start_pos=prompt_len)
        # 逐 token 前向 KL 近似：log(π/π_ref) → exp(x) - x - 1（k3 估计，非负）
        log_ratio = new_logprobs - ref_logprobs
        kl = torch.exp(log_ratio) - log_ratio - 1.0
        if completion_lengths is not None:
            kl_term = (kl * mask).sum() / mask.sum().clamp(min=1.0)
        else:
            kl_term = kl.mean()
        policy_loss = policy_loss + kl_beta * kl_term
    return policy_loss


def grpo_update_step(model, optimizer, prompt_ids, reward_fn,
                     group_size=4, max_new_tokens=32, context_length=None,
                     temperature=1.0, top_k=None, top_p=None, eos_token_id=None,
                     clip_eps=0.2, kl_beta=0.0, ref_model=None, decode_fn=None,
                     max_grad_norm: float = 1.0, use_completion_mask: bool = True):
    """单步 GRPO 更新（reasoning ch06 训练循环的原子操作）。

    流程：采样 G 条 -> reward_fn 打分 -> 组内优势 -> clipped PG loss -> 反传更新。

    Args:
        reward_fn: callable[token_ids_row, str] -> float；str 为 decode_fn 解码文本
        decode_fn: token 张量 -> 文本（供 reward_fn 使用）；None 时 reward_fn 直接收 ids
        max_grad_norm: 梯度范数裁剪（GRPO 标配 1.0；None 关闭）
        use_completion_mask: True 时 loss/KL 只统计真实生成 token（padding 不进梯度）

    Returns:
        dict(loss=..., mean_reward=..., advantages=..., rewards=...)
    """
    samples, old_logprobs, completion_lengths = generate_group_samples(
        model, prompt_ids, group_size=group_size, max_new_tokens=max_new_tokens,
        context_length=context_length, temperature=temperature,
        top_k=top_k, top_p=top_p, eos_token_id=eos_token_id,
    )
    rewards = []
    for row in samples:
        arg = decode_fn(row) if decode_fn is not None else row
        rewards.append(float(reward_fn(arg)))
    rewards = torch.tensor(rewards, device=samples.device)
    advantages = compute_group_advantages(rewards)

    optimizer.zero_grad()
    loss = compute_grpo_loss(
        model, samples, old_logprobs, advantages,
        prompt_len=prompt_ids.shape[1], clip_eps=clip_eps,
        kl_beta=kl_beta, ref_model=ref_model,
        completion_lengths=completion_lengths if use_completion_mask else None,
    )
    loss.backward()
    if max_grad_norm is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()

    return {
        "loss": float(loss.item()),
        "mean_reward": float(rewards.mean().item()),
        "advantages": advantages,
        "rewards": rewards,
    }


def grpo_update_epoch(model, optimizer, prompt_ids_list, reward_fn,
                      group_size=4, max_new_tokens=32, context_length=None,
                      temperature=1.0, top_k=None, top_p=None, eos_token_id=None,
                      clip_eps=0.2, kl_beta=0.0, ref_model=None, decode_fn=None,
                      max_grad_norm: float = 1.0):
    """一批 prompt 的 GRPO 更新（reasoning ch07 的 batched 工作流）。

    对 batch 内每个 prompt 各自采样 group、计算组内优势与 clipped loss，
    梯度累积后**统一一步** optimizer.step()（等价于以整批 prompt 为一个
    mini-batch 的更新，方差低于逐 prompt 更新）。

    Args:
        prompt_ids_list: [(1, T_i), ...] 一批 prompt 的 token ids（长度可不同）

    Returns:
        dict(loss=批内平均 loss, mean_reward=批内平均 reward,
             rewards=全部 reward, num_prompts=批大小)
    """
    optimizer.zero_grad()
    losses, all_rewards = [], []
    for prompt_ids in prompt_ids_list:
        samples, old_logprobs, completion_lengths = generate_group_samples(
            model, prompt_ids, group_size=group_size, max_new_tokens=max_new_tokens,
            context_length=context_length, temperature=temperature,
            top_k=top_k, top_p=top_p, eos_token_id=eos_token_id,
        )
        rewards = []
        for row in samples:
            arg = decode_fn(row) if decode_fn is not None else row
            rewards.append(float(reward_fn(arg)))
        rewards = torch.tensor(rewards, device=samples.device)
        advantages = compute_group_advantages(rewards)
        loss = compute_grpo_loss(
            model, samples, old_logprobs, advantages,
            prompt_len=prompt_ids.shape[1], clip_eps=clip_eps,
            kl_beta=kl_beta, ref_model=ref_model,
            completion_lengths=completion_lengths,
        )
        # 梯度累积（除以批大小，保持与单步等价的学习率语义）
        (loss / len(prompt_ids_list)).backward()
        losses.append(float(loss.item()))
        all_rewards.extend(rewards.tolist())

    if max_grad_norm is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()

    return {
        "loss": sum(losses) / len(losses),
        "mean_reward": sum(all_rewards) / len(all_rewards),
        "rewards": all_rewards,
        "num_prompts": len(prompt_ids_list),
    }


# 测试代码 main 函数
def main():
    # 用法示例见 tests/unit/test_grpo.py
    pass

if __name__ == "__main__":
    main()
