# -*- coding: utf-8 -*-

# ***************************************************
# * File        : grpo_variants.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2026-10-24
# * Version     : 1.0.102421
# * Description : GRPO 改进变体（reasoning-from-scratch ch07 "7.7 Improvements"
# *               三个脚本的移植），数学与原文一致，落在本项目 token 级 API
# *               （token_logprobs / completion_mask / generate_group_samples）上：
# *               1. deepseek_v32_style：off-policy 序列掩码 + 重加权 KL（序列级 ratio）
# *               2. olmo3_style：DAPO/Dr.GRPO 式 token 级 clip-higher + 截断 IS + 去 std 归一
# *               3. gdpo：分量归一化聚合优势 + 序列级 clipped 目标
# *               未移植项见各函数 docstring（采样侧基础设施差异）。
# * Link        : https://github.com/rasbt/reasoning-from-scratch/tree/main/ch07/03_rlvr_grpo_scripts_advanced/7_7_improvements
# * Requirement : torch
# ***************************************************

__all__ = [
    "compute_deepseek_v32_style_loss",
    "deepseek_v32_update_step",
    "compute_olmo3_style_loss",
    "olmo3_update_step",
    "compute_gdpo_advantages",
    "compute_gdpo_loss",
    "gdpo_update_step",
]

# python libraries
from pathlib import Path

import torch

from post_training.rl.core import (
    completion_mask,
    compute_group_advantages,
    generate_group_samples,
    token_logprobs,
)

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


def _seq_logprobs(logprobs, mask):
    """(G, L) token log-prob -> (G,) 序列求和（mask 排除 padding）。"""
    if mask is not None:
        return (logprobs * mask).sum(dim=1)
    return logprobs.sum(dim=1)


# ------------------------------
# 1. DeepSeek V3.2 style
# ------------------------------
def compute_deepseek_v32_style_loss(model, samples, old_logprobs, advantages, prompt_len,
                                    clip_eps=10.0, ref_model=None, kl_beta=0.02,
                                    off_policy_delta=0.1, completion_lengths=None):
    """DeepSeek V3.2 风格 GRPO 变体（序列级 ratio）。

    来源: reasoning-from-scratch
    ch07/03_rlvr_grpo_scripts_advanced/7_7_improvements/deepseek_v32_style.py

    与主线 GRPO 的差异（编号沿用原书注释）：
    - (10) off-policy 序列掩码：A<0 且 per-token 平均 (old-new) 序列 KL 超过
      off_policy_delta 的样本，优势整体置 0（丢弃过度偏离的负优势样本）；
    - (9)  重加权 KL：``kl_beta · mean( exp(new-old) · (new-ref) )``（序列级，
      以旧策略重要性比率加权的简单差分 KL）；
    - (12) 保留 GRPO 组归一化优势（advantages 由调用方按 compute_group_advantages 提供）。

    未移植：(11) top-p keep sampling mask——属于采样侧实现（原书在采样时保存
    top-p 掩码并在训练前向时屏蔽非采样子空间），本项目采样统一在 layers/sampling。
    """
    new_logprobs = token_logprobs(model, samples, start_pos=prompt_len)
    if completion_lengths is not None:
        mask = completion_mask(completion_lengths, new_logprobs.shape[1]).float()
        gen_lens = mask.sum(dim=1).clamp(min=1.0)
    else:
        mask = None
        gen_lens = torch.full_like(advantages, new_logprobs.shape[1])
    seq_new = _seq_logprobs(new_logprobs, mask)
    seq_old = _seq_logprobs(old_logprobs, mask).detach()

    ratio = torch.exp(seq_new - seq_old)
    # (10) off-policy 序列掩码
    seq_kl = (seq_old - seq_new).detach() / gen_lens
    off_policy_mask = torch.ones_like(advantages)
    off_policy_mask[(advantages < 0) & (seq_kl > off_policy_delta)] = 0
    adv = advantages * off_policy_mask

    unclipped = ratio * adv
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
    loss = -torch.minimum(unclipped, clipped).mean()

    if kl_beta > 0.0 and ref_model is not None:
        with torch.no_grad():
            ref_logprobs = token_logprobs(ref_model, samples, start_pos=prompt_len)
        seq_ref = _seq_logprobs(ref_logprobs, mask)
        # (9) 重加权 KL（旧策略重要性比率加权）
        kl_ratio = torch.exp(seq_new - seq_old)
        loss = loss + kl_beta * torch.mean(kl_ratio * (seq_new - seq_ref))
    return loss


def deepseek_v32_update_step(model, optimizer, prompt_ids, reward_fn,
                             group_size=4, max_new_tokens=32, context_length=None,
                             temperature=1.0, top_k=None, top_p=None,
                             eos_token_id=None, clip_eps=10.0, kl_beta=0.02,
                             ref_model=None, off_policy_delta=0.1, decode_fn=None,
                             max_grad_norm: float = 1.0):
    """deepseek_v32_style 的单步更新（采样/反传流程同 grpo_update_step）。"""
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
    loss = compute_deepseek_v32_style_loss(
        model, samples, old_logprobs, advantages,
        prompt_len=prompt_ids.shape[1], clip_eps=clip_eps, ref_model=ref_model,
        kl_beta=kl_beta, off_policy_delta=off_policy_delta,
        completion_lengths=completion_lengths,
    )
    loss.backward()
    if max_grad_norm is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()
    return {
        "loss": float(loss.item()),
        "mean_reward": float(rewards.mean().item()),
        "rewards": rewards,
    }


# ------------------------------
# 2. OLMo3 style (DAPO / Dr. GRPO)
# ------------------------------
def compute_olmo3_style_loss(model, samples, old_logprobs, advantages, prompt_len,
                             clip_eps=0.2, clip_eps_high=10.0, completion_lengths=None):
    """OLMo3 风格 GRPO 变体（token 级，DAPO/Dr.GRPO 改进集）。

    来源: reasoning-from-scratch
    ch07/03_rlvr_grpo_scripts_advanced/7_7_improvements/olmo3_style.py

    与主线 GRPO 的差异（编号沿用原书注释）：
    - (5) clip-higher：下界 1-clip_eps、上界 1+clip_eps_high（不对称）；
    - (6) 截断重要性采样：ratio 先 ``clamp(max=1+clip_eps_high)`` 再进目标；
    - (3) token 级目标：全部有效 token 一起平均（非逐序列平均）；
    - (7) 优势不除 std（Dr.GRPO），由调用方传入 A = r - mean(r)；
    - (4) 无 KL 项（DAPO/Dr.GRPO 默认关闭）。

    未移植：(1) 零梯度信号过滤与 (2) active sampling 属于训练循环层，
    见 olmo3_update_step 的 zero_advantage 处理。
    """
    new_logprobs = token_logprobs(model, samples, start_pos=prompt_len)
    ratio = torch.exp(new_logprobs - old_logprobs)
    ratio = torch.clamp(ratio, max=1.0 + clip_eps_high)               # (6) 截断 IS
    clipped_ratio = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps_high)  # (5) clip-higher
    adv = advantages.unsqueeze(1)
    token_obj = torch.minimum(ratio * adv, clipped_ratio * adv)
    if completion_lengths is not None:
        mask = completion_mask(completion_lengths, new_logprobs.shape[1]).float()
        return -(token_obj * mask).sum() / mask.sum().clamp(min=1.0)  # (3) token 级
    return -token_obj.mean()


def olmo3_update_step(model, optimizer, prompt_ids, reward_fn,
                      group_size=4, max_new_tokens=32, context_length=None,
                      temperature=1.0, top_k=None, top_p=None, eos_token_id=None,
                      clip_eps=0.2, clip_eps_high=10.0, decode_fn=None,
                      skip_zero_advantage: bool = True, max_grad_norm: float = 1.0):
    """olmo3_style 的单步更新。

    (1) 零梯度信号过滤：组内 reward 全相同（优势恒 0）时跳过反传/更新，
    返回 ``zero_advantage=True``（训练循环可据此换 prompt 重采样，
    即 (2) active sampling）。
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
    advantages = rewards - rewards.mean()                             # (7) 去 std 归一

    result = {
        "mean_reward": float(rewards.mean().item()),
        "rewards": rewards,
        "zero_advantage": bool(rewards.max().item() == rewards.min().item()),
    }
    if result["zero_advantage"] and skip_zero_advantage:
        result["loss"] = 0.0
        return result

    optimizer.zero_grad()
    loss = compute_olmo3_style_loss(
        model, samples, old_logprobs, advantages,
        prompt_len=prompt_ids.shape[1], clip_eps=clip_eps,
        clip_eps_high=clip_eps_high, completion_lengths=completion_lengths,
    )
    loss.backward()
    if max_grad_norm is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()
    result["loss"] = float(loss.item())
    return result


# ------------------------------
# 3. GDPO
# ------------------------------
def compute_gdpo_advantages(correctness_rewards, format_rewards,
                            format_coeff=0.1, eps=1e-4):
    """GDPO 优势：各 reward 分量组内归一化后加权聚合，再整体重新归一化。

    ``adv_c = (r_c - mean)/(std+eps)``，``adv_f`` 同理；
    ``adv_sum = adv_c + format_coeff·adv_f``；
    ``adv = (adv_sum - mean)/(std+eps)``。
    """
    rc = torch.as_tensor(correctness_rewards, dtype=torch.float32)
    rf = torch.as_tensor(format_rewards, dtype=torch.float32)
    adv_correct = (rc - rc.mean()) / (rc.std() + eps)
    adv_format = (rf - rf.mean()) / (rf.std() + eps)
    adv_sum = adv_correct + format_coeff * adv_format
    return (adv_sum - adv_sum.mean()) / (adv_sum.std() + eps)


def compute_gdpo_loss(model, samples, old_logprobs, advantages, prompt_len,
                      clip_eps=10.0, kl_beta=0.02, ref_model=None,
                      completion_lengths=None):
    """GDPO loss：序列级 clipped 目标 + ``kl_beta · mean(new - ref)``（序列求和口径）。

    来源: reasoning-from-scratch
    ch07/03_rlvr_grpo_scripts_advanced/7_7_improvements/gdpo.py
    （该脚本的损失与主线序列级 GRPO 相同；GDPO 特有的数学在优势构造，
    见 compute_gdpo_advantages 与 gdpo_update_step 的 reward 拆分。）
    """
    new_logprobs = token_logprobs(model, samples, start_pos=prompt_len)
    if completion_lengths is not None:
        mask = completion_mask(completion_lengths, new_logprobs.shape[1]).float()
    else:
        mask = None
    seq_new = _seq_logprobs(new_logprobs, mask)
    seq_old = _seq_logprobs(old_logprobs, mask).detach()
    ratio = torch.exp(seq_new - seq_old)

    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
    loss = -torch.minimum(unclipped, clipped).mean()

    if kl_beta > 0.0 and ref_model is not None:
        with torch.no_grad():
            ref_logprobs = token_logprobs(ref_model, samples, start_pos=prompt_len)
        seq_ref = _seq_logprobs(ref_logprobs, mask)
        loss = loss + kl_beta * torch.mean(seq_new - seq_ref)
    return loss


def gdpo_update_step(model, optimizer, prompt_ids, correctness_reward_fn,
                     format_reward_fn, group_size=4, max_new_tokens=32,
                     context_length=None, temperature=1.0, top_k=None, top_p=None,
                     eos_token_id=None, clip_eps=10.0, kl_beta=0.02, ref_model=None,
                     format_coeff=0.1, decode_fn=None, max_grad_norm: float = 1.0):
    """GDPO 单步更新：正确性奖励与格式奖励分别打分 -> 分量归一化聚合优势。

    原书 reward = correctness + format_coeff·format（标量相加）；GDPO 改进点
    在优势侧（分量分别归一化，见 compute_gdpo_advantages）。
    """
    samples, old_logprobs, completion_lengths = generate_group_samples(
        model, prompt_ids, group_size=group_size, max_new_tokens=max_new_tokens,
        context_length=context_length, temperature=temperature,
        top_k=top_k, top_p=top_p, eos_token_id=eos_token_id,
    )
    correct_rewards, fmt_rewards = [], []
    for row in samples:
        arg = decode_fn(row) if decode_fn is not None else row
        correct_rewards.append(float(correctness_reward_fn(arg)))
        fmt_rewards.append(float(format_reward_fn(arg)))
    correct_rewards = torch.tensor(correct_rewards, device=samples.device)
    fmt_rewards = torch.tensor(fmt_rewards, device=samples.device)
    advantages = compute_gdpo_advantages(correct_rewards, fmt_rewards,
                                         format_coeff=format_coeff)

    optimizer.zero_grad()
    loss = compute_gdpo_loss(
        model, samples, old_logprobs, advantages,
        prompt_len=prompt_ids.shape[1], clip_eps=clip_eps, kl_beta=kl_beta,
        ref_model=ref_model, completion_lengths=completion_lengths,
    )
    loss.backward()
    if max_grad_norm is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()
    return {
        "loss": float(loss.item()),
        "mean_reward": float((correct_rewards + format_coeff * fmt_rewards).mean().item()),
        "rewards": correct_rewards + format_coeff * fmt_rewards,
    }


# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
