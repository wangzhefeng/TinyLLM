# -*- coding: utf-8 -*-

# ***************************************************
# * File        : batched.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2026-10-24
# * Version     : 1.0.102421
# * Description : GRPO 批量采样与批量更新（reasoning-from-scratch ch06
# *               的 batched 工作流）：prompt 复制 G 份一次前向生成整组，
# *               通用实现（无 KV cache 的批量自回归循环）。
# * Link        : https://github.com/rasbt/reasoning-from-scratch/blob/main/ch06/02_rlvr_grpo_scripts_intro/rlvr_grpo_original_no_kl_batched.py
# * Requirement : torch
# ***************************************************

__all__ = [
    "generate_group_samples_batched",
    "grpo_update_step_batched",
]

# python libraries
from pathlib import Path

import torch

from layers.sampling import sample_next_token
from post_training.rl.core import (
    compute_group_advantages,
    compute_grpo_loss,
    token_logprobs,
)

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


@torch.no_grad()   # 不能用 inference_mode：其产出的张量无法参与后续 backward
def generate_group_samples_batched(model, prompt_ids, group_size=4, max_new_tokens=32,
                                   context_length=None, temperature=1.0, top_k=None,
                                   top_p=None, eos_token_id=None, pad_id=None):
    """批量版组采样：prompt 复制 G 份，单个自回归循环同时生成整组。

    返回结构与 :func:`post_training.rl.core.generate_group_samples` 对齐：
    ``(samples (G, T+L), old_logprobs (G, L), completion_lengths (G,))``。
    提前 EOS 的行继续用 pad token 补齐到组内最长（pad_id=None 时重复
    末 token，与串行版的 padding 完全一致；greedy 下两者输出逐位相同），
    padding 位由 completion_mask 排除。
    """
    model.eval()
    prompt_len = prompt_ids.shape[1]
    ctx_len = context_length or prompt_len + max_new_tokens
    seq = prompt_ids.expand(group_size, -1)
    finished = torch.zeros(group_size, dtype=torch.bool, device=seq.device)
    comp_lens = torch.zeros(group_size, dtype=torch.long, device=seq.device)
    for _ in range(max_new_tokens):
        logits = model(seq[:, -ctx_len:])
        next_idx = sample_next_token(logits[:, -1, :], temperature=temperature,
                                     top_k=top_k, top_p=top_p)        # (G, 1)
        if eos_token_id is not None:
            just_eos = next_idx.squeeze(1) == eos_token_id
            if pad_id is not None:
                pad_token = torch.full_like(next_idx, pad_id)
            else:
                pad_token = seq[:, -1:]                               # 重复末 token（与串行版一致）
            next_idx = torch.where(finished.unsqueeze(1), pad_token, next_idx)
        else:
            just_eos = None
        comp_lens = comp_lens + (~finished).long()
        if eos_token_id is not None:
            finished = finished | just_eos
        seq = torch.cat([seq, next_idx], dim=1)
        if eos_token_id is not None and bool(finished.all()):
            break
    samples = seq
    completion_lengths = comp_lens
    # 旧策略 log-prob（无梯度）
    old_logprobs = token_logprobs(model, samples, start_pos=prompt_len)
    return samples.clone(), old_logprobs.clone(), completion_lengths


def grpo_update_step_batched(model, optimizer, prompt_ids, reward_fn,
                             group_size=4, max_new_tokens=32, context_length=None,
                             temperature=1.0, top_k=None, top_p=None,
                             eos_token_id=None, pad_id=None,
                             clip_eps=0.2, clip_eps_high=None, kl_beta=0.0,
                             kl_estimator="k3", loss_norm="token_mean",
                             ref_model=None, decode_fn=None,
                             max_grad_norm: float = 1.0,
                             use_completion_mask: bool = True):
    """单步 GRPO 更新（批量采样版），镜像 :func:`post_training.rl.core.grpo_update_step`。

    与串行版唯一差异：组内 G 条生成由一次批量自回归循环产出。
    """
    samples, old_logprobs, completion_lengths = generate_group_samples_batched(
        model, prompt_ids, group_size=group_size, max_new_tokens=max_new_tokens,
        context_length=context_length, temperature=temperature,
        top_k=top_k, top_p=top_p, eos_token_id=eos_token_id, pad_id=pad_id,
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
        clip_eps_high=clip_eps_high, kl_beta=kl_beta, ref_model=ref_model,
        kl_estimator=kl_estimator, loss_norm=loss_norm,
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


# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
