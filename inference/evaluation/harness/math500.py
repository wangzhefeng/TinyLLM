# -*- coding: utf-8 -*-

# ***************************************************
# * File        : math500.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2026-10-24
# * Version     : 1.0.102421
# * Description : MATH-500 端到端评测 harness（reasoning-from-scratch ch04/ch05）：
# *               把"生成 -> 解析 -> 评分 -> 统计"串成评测循环，支持
# *               greedy/cot/self-consistency/weighted-sc/best-of-n/
# *               self-refinement 六种策略。generate_fn / batched_generate_fn
# *               可注入替身，评测循环完全离线可单测；重依赖
# *               （torch/loading/strategies）全部在函数内懒加载。
# * Link        : https://github.com/rasbt/reasoning-from-scratch/tree/main/ch04
# * Requirement : sympy（经 math_verifiers）；CLI 另需 torch
# ***************************************************

__all__ = [
    "evaluate_math500",
    "evaluate_math500_batched",
    "save_report",
    "heuristic_score",
    "main",
]

# python libraries
import argparse
import json
import math
import os
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)

from inference.evaluation.benchmarks.math_verifiers import (
    extract_final_candidate,
    grade_answer,
    render_prompt,
)

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]
os.environ['LOG_NAME'] = LOGGING_LABEL
from utils.log_util import logger

STRATEGIES = (
    "greedy", "cot", "self-consistency",
    "weighted-sc", "best-of-n", "self-refinement",
)


# ------------------------------
# 评分统计与落盘
# ------------------------------
def _summarize(results):
    """results -> {"accuracy", "num_problems", "results"} 评测报告。"""
    num_problems = len(results)
    num_correct = sum(int(r["correct"]) for r in results)
    return {
        "accuracy": num_correct / num_problems if num_problems else 0.0,
        "num_problems": num_problems,
        "results": results,
    }


def save_report(report, path):
    """评测报告 JSON 落盘（自动建目录）。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return path


# ------------------------------
# 生成函数构建（真实模型路径，CLI 用）
# ------------------------------
def _build_generate_fn(model, tokenizer, max_new_tokens=512, temperature=0.0,
                       top_k=None, top_p=None, context_length=None, family=None):
    """构建 generate_fn(token_ids) -> 解码后完整生成文本。

    family 给定时按注册族分派（qwen3 走 generate_qwen3 + KV cache，
    只返回新增部分）；None 时用族无关的 generate（有 reset_kv_cache
    的模型自动启用 cache）。
    """
    from inference.generate_core import _resolve_ctx_len, generate, generate_qwen3

    ctx_len = _resolve_ctx_len(model, context_length)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if family is not None:
        use_cache = family in ("gpt2", "qwen3")
        gen = generate_qwen3 if family == "qwen3" else generate
        strip_prompt = family == "qwen3"
    else:
        use_cache = hasattr(model, "reset_kv_cache")
        gen = generate
        strip_prompt = True

    def fn(token_ids):
        out = gen(model, token_ids, max_new_tokens=max_new_tokens,
                  context_length=ctx_len, temperature=temperature,
                  top_k=top_k, top_p=top_p, eos_token_id=eos_token_id,
                  use_cache=use_cache)
        new_tokens = out[:, token_ids.size(1):] if strip_prompt else out
        return tokenizer.decode(new_tokens.squeeze(0).tolist())

    return fn


def _build_batched_generate_fn(model, tokenizer, max_new_tokens=512,
                               temperature=0.0, top_k=None, top_p=None,
                               context_length=None):
    """构建 batched_generate_fn(token_ids, attn_mask, pad_id) -> [文本, ...]。

    内部走 generate_qwen3_batched_cache（右填充 + attn_mask 屏蔽 pad），
    返回与 batch 行对齐的解码文本列表。
    """
    from inference.generate_core import (
        _resolve_ctx_len,
        generate_qwen3_batched_cache,
    )

    ctx_len = _resolve_ctx_len(model, context_length)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)

    def fn(token_ids, attn_mask, pad_id):
        out = generate_qwen3_batched_cache(
            model, token_ids, max_new_tokens=max_new_tokens,
            context_length=ctx_len, temperature=temperature, top_k=top_k,
            top_p=top_p, eos_token_id=eos_token_id,
            attn_mask=attn_mask, pad_id=pad_id,
        )
        return [tokenizer.decode(row.tolist()) for row in out]

    return fn


def _model_device(model):
    try:
        return next(model.parameters()).device
    except (StopIteration, AttributeError, TypeError):
        import torch
        return torch.device("cpu")


# ------------------------------
# 评测循环
# ------------------------------
def evaluate_math500(model, tokenizer, problems, generate_fn=None,
                     extract_fn=None, grade_fn=None, max_new_tokens=512,
                     cot=False, out_path=None):
    """MATH-500 逐题评测：render_prompt -> 编码 -> generate_fn -> extract -> grade。

    Args:
        model: forward(x) 返回 (B, T, V) logits 的模型
        tokenizer: 带 encode/decode 的分词器
        problems: MATH-500 条目列表（每条含 "problem" 与 "answer"）
        generate_fn: callable[token_ids, str]（解码后的完整生成文本）；
            注入替身可完全离线评测，None 时用真实模型贪心生成
        extract_fn: 原始文本 -> 候选答案；默认 extract_final_candidate
        grade_fn: (pred, expected) -> bool；默认 grade_answer
        max_new_tokens: generate_fn 缺省时真实生成的最大新 token 数
        cot: True 时 prompt 追加思维链指令
        out_path: 非 None 时报告 JSON 落盘

    Returns:
        dict: {"accuracy", "num_problems",
               "results": [{"problem", "expected", "pred", "raw", "correct"}, ...]}
    """
    import torch

    if generate_fn is None:
        generate_fn = _build_generate_fn(model, tokenizer, max_new_tokens=max_new_tokens)
    extract_fn = extract_fn or extract_final_candidate
    grade_fn = grade_fn or grade_answer

    results = []
    for problem in problems:
        prompt = render_prompt(problem["problem"], cot=cot)
        token_ids = torch.tensor(
            tokenizer.encode(prompt), dtype=torch.long
        ).unsqueeze(0)
        raw = generate_fn(token_ids)
        pred = extract_fn(raw) if raw is not None else ""
        correct = bool(grade_fn(pred, problem["answer"]))
        results.append({
            "problem": problem["problem"],
            "expected": problem["answer"],
            "pred": pred,
            "raw": raw,
            "correct": correct,
        })

    report = _summarize(results)
    if out_path is not None:
        save_report(report, out_path)
    return report


def evaluate_math500_batched(model, tokenizer, problems, max_new_tokens=512,
                             batch_size=4, batched_generate_fn=None,
                             extract_fn=None, grade_fn=None, cot=False,
                             out_path=None):
    """MATH-500 批量评测：prompt 右填充对齐成批后整批生成再逐题评分。

    Args:
        batched_generate_fn: callable[(B, T) token_ids, attn_mask, pad_id]
            -> 与 batch 行对齐的文本列表；None 时内部用
            generate_qwen3_batched_cache（pad/attn_mask 对齐），注入替身可离线测试
        其余参数同 :func:`evaluate_math500`

    Returns:
        dict: 同 :func:`evaluate_math500`
    """
    import torch

    if batched_generate_fn is None:
        batched_generate_fn = _build_batched_generate_fn(
            model, tokenizer, max_new_tokens=max_new_tokens
        )
    extract_fn = extract_fn or extract_final_candidate
    grade_fn = grade_fn or grade_answer

    device = _model_device(model)
    pad_id = getattr(tokenizer, "pad_token_id", None)
    pad_value = pad_id if pad_id is not None else 0

    results = []
    for start in range(0, len(problems), batch_size):
        batch = problems[start:start + batch_size]
        encoded = [
            tokenizer.encode(render_prompt(p["problem"], cot=cot))
            for p in batch
        ]
        max_len = max(len(ids) for ids in encoded)
        token_ids = torch.full((len(batch), max_len), pad_value,
                               dtype=torch.long, device=device)
        attn_mask = torch.zeros((len(batch), max_len),
                                dtype=torch.bool, device=device)
        for i, ids in enumerate(encoded):
            token_ids[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
            attn_mask[i, :len(ids)] = True

        texts = batched_generate_fn(token_ids, attn_mask, pad_id)
        if len(texts) != len(batch):
            raise ValueError(
                f"batched_generate_fn 返回 {len(texts)} 条结果，期望 {len(batch)} 条"
            )

        for problem, raw in zip(batch, texts):
            pred = extract_fn(raw) if raw is not None else ""
            correct = bool(grade_fn(pred, problem["answer"]))
            results.append({
                "problem": problem["problem"],
                "expected": problem["answer"],
                "pred": pred,
                "raw": raw,
                "correct": correct,
            })

    report = _summarize(results)
    if out_path is not None:
        save_report(report, out_path)
    return report


# ------------------------------
# 加权 self-consistency 默认评分器（书 ch05 heuristic_score 移植）
# ------------------------------
def heuristic_score(answer, brevity_bonus=500.0, boxed_bonus=2.0,
                    extract_bonus=1.0, fulltext_bonus=0.0):
    """启发式答案评分：有 \\boxed 加分最高，无框退而取数字，短文本微加分。"""
    score = 0.0

    cand = extract_final_candidate(answer, fallback=None)
    if cand:
        score += boxed_bonus
    else:
        cand = extract_final_candidate(answer, fallback="number_only")
        if cand:
            score += extract_bonus
        else:
            cand = extract_final_candidate(answer, fallback="number_then_full")
            if cand:
                score += fulltext_bonus

    score += 1.5 * math.exp(-len(answer) / brevity_bonus)
    return score


# ------------------------------
# CLI
# ------------------------------
def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="MATH-500 端到端评测（生成 -> 解析 -> 评分 -> 统计）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", default="qwen3-0.6b",
                        help="MODEL_REGISTRY 键")
    parser.add_argument("--strategy", default="greedy", choices=STRATEGIES,
                        help="评测策略")
    parser.add_argument("--n", type=int, default=5,
                        help="采样条数（self-consistency/weighted-sc/best-of-n）"
                             "或最大修订轮数（self-refinement）")
    parser.add_argument("--subset", type=int, default=None,
                        help="只评测前 N 道题（默认全量 500）")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="greedy/cot 批量生成的 batch 大小")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="采样温度（greedy/cot 固定贪心，忽略此项）")
    parser.add_argument("--parser", default="basic", choices=("basic", "advanced"),
                        help="答案评分器：basic=extract_final_candidate+grade_answer，"
                             "advanced=混合 LaTeX 解析器（advanced_parser，无 antlr 时自动降级）")
    parser.add_argument("--max-new-tokens", type=int, default=512,
                        help="每个问题最大生成 token 数（书 ch03/ch04 用 2048）")
    parser.add_argument("--out", default=None,
                        help="报告 JSON 路径（默认 results/math500/<model>_<strategy>.json）")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    # 重依赖懒加载（torch/loading/strategies/数据集）
    import torch
    from layers.tokenizers.tokenization import choose_tokenizer
    from inference.loading import load_model
    from inference.registry import get_model_spec
    from inference.evaluation.benchmarks.math_verifiers import load_math500_data

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    spec = get_model_spec(args.model)
    model, config, extras = load_model(args.model, device=str(device))
    name = spec.tokenizer_model or {
        "gpt2": "tiktoken_gpt2_bpe", "llama": "llama2",
    }.get(spec.family, spec.family)
    tokenizer_kwargs = {}
    if extras.get("tokenizer_file_path"):
        tokenizer_kwargs["tokenizer_file_path"] = extras["tokenizer_file_path"]
    tokenizer = choose_tokenizer(name, **tokenizer_kwargs)

    problems = load_math500_data(verbose=False)
    if args.subset is not None:
        problems = problems[:args.subset]
    out_path = args.out or f"results/math500/{args.model}_{args.strategy}.json"

    if args.parser == "advanced":
        from inference.evaluation.benchmarks.advanced_parser import (
            parse_and_verify_answer,
        )
        grade = parse_and_verify_answer
        logger.info("advanced parser enabled (hybrid LaTeX parser, antlr 不可用时降级 basic)")
    else:
        grade = grade_answer

    if args.strategy in ("greedy", "cot"):
        cot = args.strategy == "cot"
        if spec.family == "qwen3":
            # 批量后端 generate_qwen3_batched_cache 依赖 cache=/attn_mask= 前向
            # 参数，仅 qwen3 支持该签名；其他族回退逐题生成
            report = evaluate_math500_batched(
                model, tokenizer, problems,
                max_new_tokens=args.max_new_tokens,
                batch_size=args.batch_size,
                batched_generate_fn=_build_batched_generate_fn(
                    model, tokenizer, max_new_tokens=args.max_new_tokens,
                ),
                cot=cot, out_path=out_path, grade_fn=grade,
            )
        else:
            report = evaluate_math500(
                model, tokenizer, problems,
                max_new_tokens=args.max_new_tokens,
                generate_fn=_build_generate_fn(
                    model, tokenizer, family=spec.family,
                    max_new_tokens=args.max_new_tokens,
                ),
                cot=cot, out_path=out_path, grade_fn=grade,
            )
    else:
        sample_fn = _build_generate_fn(
            model, tokenizer, temperature=args.temperature, family=spec.family,
            max_new_tokens=args.max_new_tokens,
        )
        if args.strategy == "self-consistency":
            from inference.strategies.voting_and_verification import self_consistency

            def sc_generate_fn(token_ids):
                majority, answers = self_consistency(
                    model, token_ids, num_samples=args.n,
                    extract_answer=extract_final_candidate,
                    generate_fn=sample_fn, return_all=True,
                )
                for text, answer in answers:
                    if answer == majority:
                        return text
                return answers[0][0] if answers else ""

            report = evaluate_math500(
                model, tokenizer, problems, generate_fn=sc_generate_fn,
                out_path=out_path, grade_fn=grade,
            )
        elif args.strategy == "weighted-sc":
            from inference.strategies.voting_and_verification import (
                weighted_self_consistency,
            )

            def wsc_generate_fn(token_ids):
                winner, samples = weighted_self_consistency(
                    model, token_ids, heuristic_score, num_samples=args.n,
                    extract_answer=extract_final_candidate,
                    generate_fn=sample_fn, return_all=True,
                )
                for text, _, _ in samples:
                    if extract_final_candidate(text) == winner:
                        return text
                return samples[0][0] if samples else ""

            report = evaluate_math500(
                model, tokenizer, problems, generate_fn=wsc_generate_fn,
                out_path=out_path, grade_fn=grade,
            )
        elif args.strategy == "best-of-n":
            from inference.strategies.voting_and_verification import best_of_n

            results = []
            for problem in problems:
                token_ids = torch.tensor(
                    tokenizer.encode(render_prompt(problem["problem"])),
                    dtype=torch.long,
                ).unsqueeze(0)

                def reward_fn(text, _expected=problem["answer"]):
                    return float(grade(extract_final_candidate(text), _expected))

                best_text, _ = best_of_n(
                    model, token_ids, reward_fn, num_samples=args.n,
                    generate_fn=sample_fn,
                )
                pred = extract_final_candidate(best_text)
                results.append({
                    "problem": problem["problem"],
                    "expected": problem["answer"],
                    "pred": pred,
                    "raw": best_text,
                    "correct": bool(grade(pred, problem["answer"])),
                })
            report = _summarize(results)
            save_report(report, out_path)
        else:  # self-refinement
            from inference.strategies.advance_text_generation import (
                generate_with_self_refinement,
            )

            def sr_generate_fn(token_ids):
                def refine(_draft_ids, draft_text):
                    accepted = bool(extract_final_candidate(draft_text))
                    return accepted, (None if accepted else token_ids), None

                _final, _rounds, rounds_log = generate_with_self_refinement(
                    model, token_ids, refine, max_rounds=args.n,
                    generate_fn=sample_fn,
                )
                return rounds_log[-1][0]

            report = evaluate_math500(
                model, tokenizer, problems, generate_fn=sr_generate_fn,
                out_path=out_path, grade_fn=grade,
            )

    num_correct = int(round(report["accuracy"] * report["num_problems"]))
    logger.info(
        f"MATH-500 strategy={args.strategy} model={args.model} "
        f"accuracy={report['accuracy'] * 100:.1f}% "
        f"({num_correct}/{report['num_problems']})"
    )
    logger.info(f"report saved to {out_path}")
    return report


# 测试代码 main 函数
if __name__ == "__main__":
    main()
