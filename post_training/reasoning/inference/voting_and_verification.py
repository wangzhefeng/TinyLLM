# -*- coding: utf-8 -*-

# ***************************************************
# * File        : voting_and_verification.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2026-10-20
# * Version     : 2.0.102021
# * Description : 推断时扩展（reasoning-from-scratch ch04/ch05）：
# *               1. Self-Consistency：采样 N 条答案，按规范化答案多数投票
# *               2. Best-of-N：采样 N 条，按 reward/验证器选最优
# *               组合 inference/generate_core 的采样解码，无需训练。
# * Link        : Self-Consistency https://arxiv.org/abs/2203.11171
# * Requirement : torch
# ***************************************************

__all__ = [
    "self_consistency",
    "best_of_n",
    "extract_boxed_answer",
]

# python libraries
import re
from collections import Counter
from pathlib import Path

import torch

from inference.generate_core import sample_next_token

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


def _default_decode_fn(token_ids):
    """默认解码：token 张量 -> 简单字符串（测试/调试用；生产请注入 tokenizer.decode）。"""
    return " ".join(map(str, token_ids.squeeze(0).tolist()))


def _generate_one(model, token_ids, max_new_tokens, context_length,
                  temperature, top_k, top_p, eos_token_id):
    """单条采样生成（复用 sample_next_token 的解码策略），返回完整 token 序列。"""
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


def self_consistency(model, token_ids, num_samples=8, extract_answer=None,
                     decode_fn=None, max_new_tokens=64, context_length=None,
                     temperature=0.8, top_k=None, top_p=None, eos_token_id=None,
                     return_all=False):
    """Self-Consistency（自洽性投票，reasoning ch04）。

    对同一 prompt 采样 ``num_samples`` 条生成，用 ``extract_answer``
    把每条映射为可比较的规范化答案（如 ``\\boxed{42}`` -> "42"），再多数投票。
    CoT 提示（prompt 里加 "Let's think step by step"）由调用方准备。

    Args:
        model: forward(x) 返回 (B, T, V) logits 的模型
        token_ids: (1, T) prompt token ids
        num_samples: 采样条数
        extract_answer: callable[str, object]；None 时按完整生成文本投票
        decode_fn: callable[tensor, str]；None 时用调试用默认解码

    Returns:
        return_all=False: 多数答案
        return_all=True: (多数答案, [(text, answer), ...])
    """
    decode_fn = decode_fn or _default_decode_fn
    answers = []
    for _ in range(num_samples):
        out = _generate_one(model, token_ids, max_new_tokens, context_length,
                            temperature, top_k, top_p, eos_token_id)
        text = decode_fn(out)
        answer = extract_answer(text) if extract_answer is not None else text
        answers.append((text, answer))

    counts = Counter(a for _, a in answers)
    majority = counts.most_common(1)[0][0]
    if return_all:
        return majority, answers
    return majority


def best_of_n(model, token_ids, reward_fn, num_samples=8, decode_fn=None,
              max_new_tokens=64, context_length=None, temperature=0.8,
              top_k=None, top_p=None, eos_token_id=None, return_all=False):
    """Best-of-N（reasoning ch05）：采样 N 条，按 ``reward_fn(text) -> float`` 选最高分。

    与 self_consistency 的区别：BoN 需要外部 reward/验证器（如数学答案比对），
    适合有客观标准的任务；self-consistency 只需答案可规范化。

    Returns:
        return_all=False: (best_text, best_reward)
        return_all=True: (best_text, best_reward, [(text, reward), ...])
    """
    decode_fn = decode_fn or _default_decode_fn
    scored = []
    for _ in range(num_samples):
        out = _generate_one(model, token_ids, max_new_tokens, context_length,
                            temperature, top_k, top_p, eos_token_id)
        text = decode_fn(out)
        scored.append((text, float(reward_fn(text))))

    scored.sort(key=lambda x: x[1], reverse=True)
    best_text, best_reward = scored[0]
    if return_all:
        return best_text, best_reward, scored
    return best_text, best_reward


def extract_boxed_answer(text: str):
    r"""reasoning 书常用的 \\boxed{...} 答案抽取；无 boxed 时取最后一个数字。"""
    m = re.findall(r"\\boxed\{([^{}]+)\}", text)
    if m:
        return m[-1].strip()
    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    return nums[-1] if nums else None


# 测试代码 main 函数
def main():
    # 用法示例见 tests/unit/test_inference_time_scaling.py
    pass

if __name__ == "__main__":
    main()
