# -*- coding: utf-8 -*-

# ***************************************************
# * File        : distill_reasoning.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2026-10-21
# * Version     : 1.0.102121
# * Description : reasoning-trace 蒸馏 from scratch（reasoning-from-scratch ch08
# *               的主线：不是 Hinton 式 logit 蒸馏（见 distillation.py），
# *               而是"教师生成推理轨迹 -> 验证器过滤 -> 学生 SFT"）。
# *               1. generate_reasoning_traces：教师对每个 prompt 采样，
# *                  可选用 verifier 过滤，产出 (prompt, completion) 数据集
# *               2. reasoning_sft_loss：只对 completion 部分计算 CE 的 SFT 损失
# *               3. distill_sft_step：学生单步更新
# *               4. distill_sft_epoch：批量（右填充）蒸馏 SFT epoch，
# *                  reasoning_sft_loss 的向量化版本（ch08 distill_batched）
# * Link        : reasoning-from-scratch ch08
# *               Distilling Step-by-Step https://arxiv.org/abs/2305.10601
# * Requirement : torch
# ***************************************************

__all__ = [
    "generate_reasoning_traces",
    "reasoning_sft_loss",
    "distill_sft_step",
    "distill_sft_epoch",
]

# python libraries
from pathlib import Path

import torch
import torch.nn.functional as F

from post_training.rl import generate_group_samples

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]
IGNORE_INDEX = -100


def generate_reasoning_traces(teacher_model, prompt_ids, verifier_fn=None,
                              num_samples: int = 4, max_new_tokens: int = 64,
                              context_length=None, temperature=1.0,
                              top_k=None, top_p=None, eos_token_id=None,
                              decode_fn=None):
    """教师模型生成推理轨迹并（可选）验证过滤（ch08 数据构造）。

    对单个 prompt 采样 ``num_samples`` 条轨迹；提供 ``verifier_fn`` 时只保留
    验证通过的轨迹（返回值 > 0 或布尔 True），即"rejection sampling 蒸馏"。

    Args:
        teacher_model: 冻结的教师（forward(x) -> (B, T, V)）
        prompt_ids: (1, T) prompt token ids
        verifier_fn: callable[token_ids_row 或 str] -> bool/float；
            None 时不过滤（保留全部轨迹）
        decode_fn: token 张量 -> 文本（供 verifier 用）

    Returns:
        list of dict(prompt_ids=(1,T) tensor, completion_ids=(1,L) tensor,
                     reward=float)；无轨迹通过验证时为空列表
    """
    samples, _, completion_lengths = generate_group_samples(
        teacher_model, prompt_ids, group_size=num_samples,
        max_new_tokens=max_new_tokens, context_length=context_length,
        temperature=temperature, top_k=top_k, top_p=top_p,
        eos_token_id=eos_token_id,
    )
    prompt_len = prompt_ids.shape[1]
    traces = []
    for row, comp_len in zip(samples, completion_lengths):
        comp_len = int(comp_len.item())
        completion = row[prompt_len:prompt_len + comp_len].unsqueeze(0)
        arg = decode_fn(row) if decode_fn is not None else row
        reward = float(verifier_fn(arg)) if verifier_fn is not None else 1.0
        if reward > 0:
            traces.append({
                "prompt_ids": prompt_ids,
                "completion_ids": completion,
                "reward": reward,
            })
    return traces


def reasoning_sft_loss(model, prompt_ids, completion_ids):
    """只对 completion 部分计算交叉熵的 SFT 损失（prompt masked）。

    等价于 targets 中 prompt 位置全部置 -100 的标准语言模型微调：
    模型输入 = prompt + completion，损失只统计 completion 的每个 token
    （首个 completion token 由 prompt 的最后位置预测）。

    Args:
        model: 学生模型（forward(x) -> (B, T, V)）
        prompt_ids: (1, T)
        completion_ids: (1, L)

    Returns:
        标量 loss
    """
    seq = torch.cat([prompt_ids, completion_ids], dim=1)          # (1, T+L)
    logits = model(seq)[:, :-1, :]                                 # (1, T+L-1, V)
    targets = seq[:, 1:]                                           # (1, T+L-1)
    L = completion_ids.shape[1]
    # 只取 completion 对应的预测位：最后 L 个位置
    logits = logits[:, -L:, :]
    targets = targets[:, -L:]
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                           targets.reshape(-1))


def distill_sft_step(student, optimizer, prompt_ids, completion_ids,
                     max_grad_norm: float = 1.0):
    """学生在单条教师轨迹上做一步 SFT（ch08 蒸馏训练循环的原子操作）。

    Returns:
        dict(loss=...)
    """
    student.train()
    optimizer.zero_grad()
    loss = reasoning_sft_loss(student, prompt_ids, completion_ids)
    loss.backward()
    if max_grad_norm is not None:
        torch.nn.utils.clip_grad_norm_(student.parameters(), max_grad_norm)
    optimizer.step()
    return {"loss": float(loss.item())}


def _batched_reasoning_sft_loss(model, batch, pad_id=None):
    """批量 completion-only CE（右填充；reasoning_sft_loss 的向量化版本）。

    batch 为 (prompt_ids, completion_ids) 列表；右填充保证真实位置
    （因果掩码）不受 pad 影响，pad/completion 之外的 target 全部置
    ``IGNORE_INDEX``。损失按样本内 completion token 均值后再对 batch 取均值。
    """
    seqs = [torch.cat([p, c], dim=1).squeeze(0) for p, c in batch]
    prompt_lens = [p.shape[1] for p, _ in batch]
    total_lens = [s.shape[0] for s in seqs]
    max_len = max(total_lens)
    if pad_id is None and any(length < max_len for length in total_lens):
        raise ValueError("变长序列批处理需要 pad_id（右填充）")
    pad = 0 if pad_id is None else pad_id
    padded = torch.stack([
        F.pad(seq, (0, max_len - seq.shape[0]), value=pad) for seq in seqs
    ])                                                              # (B, S)
    logits = model(padded[:, :-1])                                  # (B, S-1, V)
    targets = padded[:, 1:]                                         # (B, S-1)
    ignore = torch.full_like(targets, IGNORE_INDEX)
    mask = torch.zeros_like(targets, dtype=torch.bool)
    for row, (prompt_len, total_len) in enumerate(zip(prompt_lens, total_lens)):
        # completion 预测位：prompt 最后位置预测首个 completion token
        mask[row, max(prompt_len - 1, 0):max(total_len - 1, 0)] = True
    targets = torch.where(mask, targets, ignore)
    per_token = F.cross_entropy(
        logits.transpose(1, 2).float(), targets,
        ignore_index=IGNORE_INDEX, reduction="none",
    )
    active = targets.ne(IGNORE_INDEX)
    per_example = (per_token * active).sum(dim=1) / active.sum(dim=1).clamp(min=1)
    return per_example.mean()


def distill_sft_epoch(dataset, student, optimizer, batch_size: int = 4,
                      pad_id=None, max_grad_norm: float = 1.0):
    """批量蒸馏 SFT：在 (prompt_ids, completion_ids) 数据集上跑一个 epoch。

    ch08 distill_batched 的工作循环：右填充组 batch、零梯度、
    批量 completion-only CE、可选梯度裁剪、优化器步进。

    Args:
        dataset: list of (prompt_ids=(1,T), completion_ids=(1,L))
        student: 学生模型（forward(x) -> (B, T, V)）
        pad_id: 填充 token id；批内序列等长时可为 None

    Returns:
        epoch 平均 loss（按样本数加权）；空数据集返回 0.0
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0.")
    student.train()
    total_loss = 0.0
    seen = 0
    for start in range(0, len(dataset), batch_size):
        batch = dataset[start:start + batch_size]
        optimizer.zero_grad()
        loss = _batched_reasoning_sft_loss(student, batch, pad_id=pad_id)
        loss.backward()
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_grad_norm)
        optimizer.step()
        total_loss += float(loss.item()) * len(batch)
        seen += len(batch)
    return total_loss / seen if seen else 0.0


# 测试代码 main 函数
def main():
    # 用法示例见 tests/unit/test_distillation.py
    pass

if __name__ == "__main__":
    main()
