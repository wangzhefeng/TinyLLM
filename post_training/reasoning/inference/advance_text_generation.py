# -*- coding: utf-8 -*-

# ***************************************************
# * File        : advance_text_generation.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2026-10-20
# * Version     : 2.0.102021
# * Description : Self-Refinement（reasoning-from-scratch ch05）：
# *               生成 -> （外部/模型）反馈 -> 修订 的迭代式推断时扩展。
# *               refine_fn 决定反馈来源：可以是规则验证器、外部 LLM，
# *               或同一模型的 critique prompt（模型自评）。
# * Link        : Self-Refine https://arxiv.org/abs/2303.17651
# * Requirement : torch
# ***************************************************

__all__ = [
    "generate_with_self_refinement",
    "refine_until_stable",
]

# python libraries
from pathlib import Path

import torch

from inference.generate_core import sample_next_token

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


def _sample_sequence(model, token_ids, max_new_tokens, context_length,
                     temperature, top_k, top_p, eos_token_id):
    """贪心/采样生成一条序列（无 cache 教学版）。"""
    model.eval()
    ctx_len = context_length or getattr(getattr(model, "cfg", None), "context_length", None) \
        or getattr(model, "pos_embed", None).num_embeddings
    with torch.inference_mode():
        seq = token_ids
        for _ in range(max_new_tokens):
            logits = model(seq[:, -ctx_len:])
            next_idx = sample_next_token(logits[:, -1, :], temperature=temperature,
                                         top_k=top_k, top_p=top_p)
            seq = torch.cat([seq, next_idx], dim=1)
            if eos_token_id is not None and bool((next_idx == eos_token_id).all()):
                break
    return seq


def generate_with_self_refinement(model, token_ids, refine_fn,
                                  max_new_tokens=64, max_rounds=3,
                                  context_length=None, temperature=0.7,
                                  top_k=None, top_p=None, eos_token_id=None,
                                  decode_fn=None):
    """Self-Refinement 主循环（reasoning ch05）。

    流程：生成初稿 -> ``refine_fn(draft) -> (feedback, revised)`` -> 用修订稿
    重新 prompt 模型继续修订，直到 refine_fn 判定通过或轮数用尽。

    Args:
        model: forward(x) 返回 (B, T, V) logits 的模型
        token_ids: (1, T) prompt
        refine_fn: callable[draft_token_ids, draft_text] ->
            (accepted: bool, feedback_prompt_ids or None, revised_text or None)。
            - accepted=True：通过，结束迭代
            - feedback_prompt_ids：下一轮的（prompt+反馈）token ids；
              为 None 时直接结束并返回当前稿
        decode_fn: token 张量 -> 文本（调试默认为 join ids）

    Returns:
        (final_token_ids, rounds_used, [每轮 (text, accepted)])
    """
    decode_fn = decode_fn or (lambda ids: " ".join(map(str, ids.squeeze(0).tolist())))
    rounds_log = []
    current = token_ids
    used = 0
    for used in range(1, max_rounds + 1):
        draft = _sample_sequence(model, current, max_new_tokens, context_length,
                                 temperature, top_k, top_p, eos_token_id)
        draft_text = decode_fn(draft)
        accepted, next_prompt, revised = refine_fn(draft, draft_text)
        rounds_log.append((draft_text, accepted))
        if accepted:
            return draft, used, rounds_log
        if next_prompt is None:
            return draft, used, rounds_log
        current = next_prompt
    return draft, used, rounds_log


def refine_until_stable(model, token_ids, max_new_tokens=64, max_rounds=4,
                        stable_threshold: int = 2, context_length=None,
                        temperature=0.7, top_k=None, top_p=None,
                        eos_token_id=None, decode_fn=None, extract_answer=None):
    """连续 N 轮答案不变则停止的 self-refinement 变体。

    ``extract_answer(text) -> answer`` 规范化每轮答案；每轮**从同一 prompt
    重新生成**（重问式细化），连续 ``stable_threshold`` 轮答案相同时提前结束
    （推断预算控制的教学示例，退化为 greedy 时即 self-consistency 的一致性检验）。

    Returns:
        (final_token_ids, rounds_used, [每轮答案])
    """
    decode_fn = decode_fn or (lambda ids: " ".join(map(str, ids.squeeze(0).tolist())))
    answers = []
    draft = None
    for used in range(1, max_rounds + 1):
        draft = _sample_sequence(model, token_ids, max_new_tokens, context_length,
                                 temperature, top_k, top_p, eos_token_id)
        text = decode_fn(draft)
        answers.append(extract_answer(text) if extract_answer else text)
        if len(answers) >= stable_threshold and \
                len(set(answers[-stable_threshold:])) == 1:
            return draft, used, answers
    return draft, max_rounds, answers


# 测试代码 main 函数
def main():
    # 用法示例见 tests/unit/test_inference_time_scaling.py
    pass

if __name__ == "__main__":
    main()
