# -*- coding: utf-8 -*-

# ***************************************************
# * File        : multiple_choice.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-10-05
# * Version     : 2.0.102101
# * Description : 多项选择基准评测（reasoning-from-scratch chF/02_mmlu）：
# *               1. Method 1（字母匹配）：生成后逐字符找 A/B/C/D（书实测最弱）
# *               2. Method 2（first-token logprob）：比较 prompt 后第一个新
# *                  token 位置各选项字母的 log-softmax 值（推荐）
# *               3. evaluate_mmlu 评测循环 + 随机猜测二项基线
# *               模型/数据集依赖一律延迟到函数内（import 零副作用，可单测）。
# * Link        : https://github.com/rasbt/reasoning-from-scratch/tree/main/chF/02_mmlu
# * Requirement : torch
# ***************************************************

__all__ = [
    "format_prompt",
    "get_prompt",
    "predict_choice",
    "common_prefix_len",
    "first_new_token_id",
    "predict_choice_logprob",
    "evaluate_mmlu",
    "random_guess_baseline",
    "gold_letter",
]

# python libraries
import os
import random
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)
import warnings
warnings.filterwarnings("ignore")

import torch

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]
os.environ['LOG_NAME'] = LOGGING_LABEL
from utils.log_util import logger


def load_data(verbose=False):
    """加载 MMLU 数据集（需联网；仅此函数依赖 datasets）。"""
    from datasets import (
        load_dataset,
        get_dataset_config_names,
    )

    configs = get_dataset_config_names("cais/mmlu")
    dataset = load_dataset("cais/mmlu", "high_school_mathematics")

    # Inspect the first example from the test set
    if verbose:
        example = dataset["test"][0]
        logger.info(f"first example: {example}")

    return configs, dataset


def format_prompt(example):
    """
    MMLU zero-shot prompt（书 chF/02 的主 notebook 版本）：
    尾部 "Answer: " 的空格鼓励模型下一个 token 直接输出单个字母。
    """
    prompt = (
        f"{example['question']}\n"
        f"A. {example['choices'][0]}\n"
        f"B. {example['choices'][1]}\n"
        f"C. {example['choices'][2]}\n"
        f"D. {example['choices'][3]}\n"
        "Answer: "  # Trailing space in "Answer: " encourages a single-letter next token
    )

    return prompt


def get_prompt(prompt, tokenizer, device):
    prompt_ids = tokenizer.encode(prompt)
    prompt_fmt = torch.tensor(prompt_ids, device=device)
    # Add batch dimension
    prompt_fmt = prompt_fmt.unsqueeze(0)

    return prompt_fmt


# ------------------------------
# Method 1: 字母匹配（书 chF/02 的 1_letter_matching.py，实测最弱）
# ------------------------------
def predict_choice(model, tokenizer, prompt_fmt, max_new_tokens=8):
    """生成后逐字符扫描找 A/B/C/D（Method 1）。

    依赖 qwen3 checkpoint 与 generate_qwen3_stream，故在函数内延迟 import。
    """
    from post_training.reasoning.inference.llm_basic.qwen3_06B import QWEN3_CONFIG
    from inference.generate_core import generate_qwen3_stream

    pred = None
    for t in generate_qwen3_stream(
        model=model,
        token_ids=prompt_fmt,
        max_new_tokens=max_new_tokens,
        context_length=QWEN3_CONFIG.context_length,
        temperature=0.0,
        top_k=None,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
    ):
        answer = tokenizer.decode(t.squeeze(0).tolist())
        for letter in answer:
            letter = letter.upper()
            # stop as soon as a letter appears
            if letter in "ABCD":
                pred = letter
                break
        if pred:
            break

    return pred


# ------------------------------
# Method 2: first-token logprob（书 chF/02 的 2_logprob.py，推荐）
# ------------------------------
def common_prefix_len(a, b):
    """两个 token id 序列公共前缀长度（书 chF/02 原版辅助函数）。"""
    i = 0
    n = min(len(a), len(b))
    while i < n and a[i] == b[i]:
        i += 1
    return i


def first_new_token_id(tokenizer, prompt, prompt_ids, continuation):
    """对 ``prompt + continuation`` 编码，找出 continuation 产生的第一个新 token id。

    处理 BPE 合并差异：不能直接假设字母的 token id 与接在 prompt 后的相同，
    需要先编码完整文本再用公共前缀对齐（书 chF/02 原版做法）。
    """
    ids_full = tokenizer.encode(prompt + continuation)
    j = common_prefix_len(ids_full, prompt_ids)
    if j >= len(ids_full):
        raise ValueError("Continuation produced no new tokens.")
    return ids_full[j]


def predict_choice_logprob(model, tokenizer, prompt_token_ids,
                           prompt=None, prompt_ids=None, letters="ABCD"):
    """First-token logprob 法（Method 2，书 chF/02 的 2_logprob.py）。

    取模型在 prompt 后**第一个新 token 位置**的 log-softmax，比较各选项
    字母（经 tokenizer 编码对齐）的 log-prob，取最大者为预测选项。
    无需生成、无需采样，比 Method 1 稳定（书实测 34% vs 21%）。

    Args:
        model: forward(x) 返回 (B, T, V) logits 的模型
        tokenizer: 带 encode/decode 的分词器
        prompt_token_ids: (1, T) prompt token 张量
        prompt: prompt 原文；None 时用 tokenizer.decode(prompt_ids) 还原
        prompt_ids: prompt id 列表；None 时取 prompt_token_ids[0].tolist()
        letters: 候选字母序列（默认 MMLU 四选一）

    Returns:
        (pred_letter, scores)：scores 为 {字母: logprob} 字典
    """
    if prompt_ids is None:
        prompt_ids = prompt_token_ids[0].tolist()
    if prompt is None:
        prompt = tokenizer.decode(prompt_ids)

    with torch.no_grad():
        logits = model(prompt_token_ids)          # [1, T, V]
        next_logp = torch.log_softmax(logits[0, -1], dim=-1)

    scores = {}
    for letter in letters:
        tok_id = first_new_token_id(tokenizer, prompt, prompt_ids, letter)
        scores[letter] = next_logp[tok_id].item()

    pred = max(scores, key=scores.get)
    return pred, scores


# ------------------------------
# MMLU 评测循环
# ------------------------------
def gold_letter(ans, letters="ABCD"):
    """MMLU 的 gold（正确答案）归一化：int -> 字母；str -> 大写首字母。"""
    if isinstance(ans, int):
        return letters[ans]
    s = str(ans).strip().upper()
    return s if s in letters else s[:1]


def evaluate_mmlu(model, tokenizer, data, predict_fn, max_examples=None,
                  verbose_every=50):
    """MMLU 风格评测循环（书 chF/02 evaluate_mmlu_* 的通用版）。

    Args:
        model: forward(x) 返回 (B, T, V) logits 的模型
        tokenizer: 带 encode/decode 的分词器
        data: MMLU 格式条目迭代器（question/choices/answer；answer 兼容 int 与 str）
        predict_fn: callable(model, tokenizer, prompt_fmt) -> 预测字母，
            例如 Method 1 的 predict_choice，或包装 predict_choice_logprob 的闭包
        max_examples: 最多评多少条（None = 全量）
        verbose_every: 每多少条打印一次进度

    Returns:
        dict: {"accuracy", "num_correct", "num_examples"}
    """
    device = next(model.parameters()).device
    total = 0
    correct = 0

    for example in data:
        if max_examples is not None and total >= max_examples:
            break
        prompt = format_prompt(example)
        prompt_fmt = get_prompt(prompt, tokenizer, device)
        pred = predict_fn(model, tokenizer, prompt_fmt)
        gold = gold_letter(example["answer"])

        total += 1
        correct += int(pred == gold)

        if verbose_every and total % verbose_every == 0:
            logger.info(f"MMLU {total} acc={correct / total:.3f}")

    accuracy = correct / max(1, total)
    return {
        "accuracy": accuracy,
        "num_correct": correct,
        "num_examples": total,
    }


# ------------------------------
# 随机猜测基线（书 chF/02 的 random_guessing_baseline.py）
# ------------------------------
def random_guess_baseline(num_questions, num_choices=4, seed=123):
    """随机猜测基线：均匀从 num_choices 个选项中猜。

    期望准确率为 1/num_choices（二项分布的均值）；同时给出一次固定种子
    随机序列的实测准确率，用于与模型评测结果对照（书 chF/02 的
    random_guessing_baseline.py 主题的最小实现）。

    Args:
        num_questions: 题目数量
        num_choices: 选项数（默认 4）
        seed: 随机种子（可复现）

    Returns:
        dict: {"expected", "empirical", "num_questions", "num_choices", "seed"}
    """
    letters = "".join(chr(ord("A") + i) for i in range(num_choices))
    rng = random.Random(seed)
    guesses = [rng.choice(letters) for _ in range(num_questions)]
    # 基线无真实标签时，gold 也按均匀分布生成（期望收敛到 1/num_choices）
    golds = [rng.choice(letters) for _ in range(num_questions)]
    empirical = sum(g == y for g, y in zip(guesses, golds)) / max(1, num_questions)

    return {
        "expected": 1.0 / num_choices,
        "empirical": empirical,
        "num_questions": num_questions,
        "num_choices": num_choices,
        "seed": seed,
    }



# 测试代码 main 函数
def main():
    """离线演示：tiny GPT-2 + 字符级 mock tokenizer 跑通 Method 2 与评测循环。

    Method 1（predict_choice）需要 qwen3 checkpoint，此处不演示。
    """
    from types import SimpleNamespace

    from models.gpt2.gpt2_124M import Model

    class _MockTokenizer:
        """字符级 mock 分词器：encode 按字符映射 id，decode 还原。"""

        def __init__(self, vocab_size=32):
            self.vocab_size = vocab_size

        def encode(self, text):
            return [ord(ch) % self.vocab_size for ch in text]

        def decode(self, ids):
            if isinstance(ids, torch.Tensor):
                ids = ids.squeeze(0).tolist() if ids.dim() == 2 else ids.tolist()
            return "".join(chr(i) for i in ids)

    # tiny GPT-2（未训练，仅演示 API 机制）
    torch.manual_seed(123)
    cfg = SimpleNamespace(
        vocab_size=32, context_length=128, embed_dim=32, d_ff=64,
        n_heads=4, n_layers=2, dropout=0.0, qkv_bias=True,
        dtype=torch.float32, is_train=False,
    )
    model = Model(cfg).eval()
    tokenizer = _MockTokenizer(vocab_size=32)

    # 演示数据：两道 MMLU 风格的题（answer 兼容 int 与 str）
    data = [
        {"question": "1 + 1 = ?", "choices": ["1", "2", "3", "4"], "answer": 1},
        {"question": "2 * 3 = ?", "choices": ["5", "6", "7", "8"], "answer": "B"},
    ]

    def predict_fn(model_, tokenizer_, prompt_fmt):
        pred, scores = predict_choice_logprob(
            model_, tokenizer_, prompt_fmt, letters="ABCD",
        )
        return pred

    results = evaluate_mmlu(
        model, tokenizer, data, predict_fn,
        max_examples=None, verbose_every=0,
    )
    logger.info(f"Method 2 (logprob) on mock data: {results}")

    baseline = random_guess_baseline(num_questions=100, num_choices=4, seed=123)
    logger.info(f"random guess baseline: {baseline}")


if __name__ == "__main__":
    main()
