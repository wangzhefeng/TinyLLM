# -*- coding: utf-8 -*-

# ***************************************************
# * File        : data_gen.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2026-10-24
# * Version     : 1.0.102421
# * Description : ch08 蒸馏教师数据生成入口：MATH-500 训练子集 ->
# *               教师生成推理轨迹（local / ollama / openrouter 三后端）->
# *               verifier 过滤 -> verified JSONL 落盘 ->
# *               average_field_lengths 字段长度统计。
# *               重依赖（torch/模型权重/API client）全部懒加载。
# * Link        : https://github.com/rasbt/reasoning-from-scratch/tree/main/ch08/02_generate_distillation_data
# * Requirement : torch（local 后端）；ollama/openai SDK 按后端需要
# ***************************************************

__all__ = [
    "generate_teacher_dataset",
    "average_field_lengths",
]

# python libraries
import argparse
import json
import random
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)

from utils.log_util import logger

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]

DEFAULT_OUT_PATH = "dataset/finetuning/math_distill/train.jsonl"
DEFAULT_MODEL_NAMES = {
    "ollama": "qwen3:30b-thinking",
    "openrouter": "deepseek/deepseek-r1",
}


def build_tokenizer(model_key, extras):
    """按模型族构造 tokenizer（qwen3/gemma3/phi 需要 tokenizer_file_path）。"""
    from models.registry import get_model_spec
    from layers.tokenizers.tokenization import choose_tokenizer

    family = get_model_spec(model_key).family
    if "tokenizer_file_path" in extras:
        return choose_tokenizer(family, tokenizer_file_path=extras["tokenizer_file_path"])
    return choose_tokenizer({
        "gpt2": "tiktoken_gpt2_bpe",
        "llama": "llama2",
    }.get(family, family))


def load_records(json_path):
    """读取 JSON 数组或 JSONL 记录（average_field_lengths 用）。"""
    path = Path(json_path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, dict):
        return parsed.get("records", [parsed])
    raise ValueError(f"Unsupported JSON root type in {path}: {type(parsed).__name__}")


def _load_problems(train_size, seed):
    """MATH-500 训练子集（与 post_training/rl/run.py 同口径的 seed 抽样）。"""
    from evaluation.verifiers.math import load_math500_data

    math_data = load_math500_data(verbose=False)
    rng = random.Random(seed)
    return rng.sample(math_data, train_size) \
        if train_size < len(math_data) else list(math_data)


def _strip_think_tags(text):
    return str(text).replace("<think>", "").replace("</think>", "").strip()


def _verified_completion(completion, expected):
    """抽取最终候选答案并做数学等价验证。"""
    from evaluation.verifiers.math import (
        extract_final_candidate,
        grade_answer,
    )

    completion = _strip_think_tags(completion)
    candidate = extract_final_candidate(completion)
    verified = bool(grade_answer(candidate, expected)) if completion else False
    return completion, verified


def _generate_local(problem, prompt, teacher_model, tokenizer,
                    num_samples, max_new_tokens):
    """local 后端：教师模型采样 + verifier 过滤（generate_reasoning_traces）。"""
    import torch

    from evaluation.verifiers.math import (
        extract_final_candidate,
        grade_answer,
    )
    from post_training.reasoning.training.distill_reasoning import (
        generate_reasoning_traces,
    )

    prompt_ids = torch.tensor(
        tokenizer.encode(prompt), dtype=torch.long,
    ).unsqueeze(0)
    expected = problem["answer"]
    eos_token_id = getattr(tokenizer, "eos_token_id", None)

    def decode_fn(row, prompt_len=prompt_ids.shape[1]):
        return tokenizer.decode(row[prompt_len:].tolist())

    def verifier_fn(text, expected=expected):
        return grade_answer(extract_final_candidate(text), expected)

    traces = generate_reasoning_traces(
        teacher_model, prompt_ids, verifier_fn=verifier_fn,
        num_samples=num_samples, max_new_tokens=max_new_tokens,
        eos_token_id=eos_token_id, decode_fn=decode_fn,
    )
    return [
        tokenizer.decode(trace["completion_ids"].squeeze(0).tolist())
        for trace in traces
    ]


def generate_teacher_dataset(problems=None, backend="local", model_name=None,
                             num_samples=4, max_new_tokens=2048,
                             out_path=DEFAULT_OUT_PATH, teacher_model=None,
                             tokenizer=None, seed=42, num_ctx=2048,
                             train_size=50):
    """生成蒸馏教师数据集（ch08 02_generate_distillation_data 工程化）。

    Args:
        problems: 题目列表（dict 含 problem/answer）；None 时懒加载
            MATH-500 训练子集（--train-size seed 抽样，与 rl/run.py 口径一致）
        backend: "local"（教师模型+tokenizer 采样+verifier 过滤）/
            "ollama"（evaluation.clients.ollama.query_model）/
            "openrouter"（evaluation.clients.openrouter 新客户端）
        model_name: ollama/openrouter 后端的教师模型名；
            local 后端为 MODEL_REGISTRY 键（配合 teacher_model/tokenizer）
        num_samples: local 后端每题采样轨迹数（API 后端每题 1 条）
        out_path: verified 记录 JSONL 输出路径
        train_size: problems 为 None 时的抽样题数

    Returns:
        list[dict]：全部记录 {"problem", "prompt", "completion",
        "expected", "verified": bool}；仅 verified 的写入 out_path
    """
    if problems is None:
        problems = _load_problems(train_size, seed)
    if backend == "local":
        if teacher_model is None or tokenizer is None:
            raise ValueError("local 后端需要传入 teacher_model 与 tokenizer")
    else:
        if backend not in DEFAULT_MODEL_NAMES:
            raise ValueError(f"backend must be one of {sorted(DEFAULT_MODEL_NAMES | {'local'})}")
        model_name = model_name or DEFAULT_MODEL_NAMES[backend]
        if backend == "openrouter":
            from evaluation.clients.openrouter import create_client

            client = create_client()

    from evaluation.verifiers.math import render_prompt

    records = []
    verified_count = 0
    for idx, problem in enumerate(problems, start=1):
        prompt = render_prompt(problem["problem"])
        if backend == "local":
            completions = _generate_local(
                problem, prompt, teacher_model, tokenizer,
                num_samples, max_new_tokens,
            )
        elif backend == "ollama":
            from evaluation.clients.ollama import query_model

            completions = [query_model(prompt, model=model_name, num_ctx=num_ctx)]
        else:
            from evaluation.clients.openrouter import run_openrouter

            completions = [run_openrouter(
                prompt, client, model=model_name, max_tokens=max_new_tokens,
            )]

        for completion in completions:
            completion, verified = _verified_completion(completion, problem["answer"])
            verified_count += int(verified)
            records.append({
                "problem": problem["problem"],
                "prompt": prompt,
                "completion": completion,
                "expected": problem["answer"],
                "verified": verified,
            })
        logger.info(
            f"[data_gen {idx}/{len(problems)}] backend={backend} "
            f"records={len(records)} verified={verified_count}"
        )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for record in records:
            if record["verified"]:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
    logger.info(
        f"Generated {len(records)} records ({verified_count} verified) "
        f"-> {out_path}（仅 verified 落盘）"
    )
    return records


def _value_to_text(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def average_field_lengths(json_path, tokenizer=None):
    """按字段统计 token 长度（avg/min/max/count），参考书 average_field_lengths_json.py。

    tokenizer 缺省时抛错提示传入（如 choose_tokenizer("qwen3",
    tokenizer_file_path=...) 构造），不做隐式下载。

    Returns:
        list[dict]: 每字段 {field, avg_tokens, min_tokens, max_tokens, count}
    """
    if tokenizer is None:
        raise ValueError(
            "需要 tokenizer 参数（如 layers.tokenizers.tokenization."
            "choose_tokenizer 构造），用于统计字段 token 长度"
        )
    records = load_records(json_path)
    rows = []
    fields = sorted({field for record in records for field in record})
    for field in fields:
        lengths = [
            len(tokenizer.encode(_value_to_text(record[field])))
            for record in records if field in record
        ]
        rows.append({
            "field": field,
            "avg_tokens": sum(lengths) / len(lengths) if lengths else 0.0,
            "min_tokens": min(lengths) if lengths else 0,
            "max_tokens": max(lengths) if lengths else 0,
            "count": len(lengths),
        })
    return rows


def args_parse():
    parser = argparse.ArgumentParser(
        description="TinyLLM ch08 蒸馏教师数据生成（local/ollama/openrouter）",
    )
    parser.add_argument("--backend", type=str, default="local",
                        choices=["local", "ollama", "openrouter"])
    parser.add_argument("--model", type=str, required=False, default=None,
                        help="教师模型：local 为 MODEL_REGISTRY 键，"
                             "ollama/openrouter 为服务端模型名（缺省取默认）")
    parser.add_argument("--train-size", "--train_size", type=int, required=False,
                        default=50, help="MATH-500 训练子集题数（seed 抽样）")
    parser.add_argument("--num-samples", "--num_samples", type=int, required=False,
                        default=4, help="local 后端每题采样轨迹数")
    parser.add_argument("--max-new-tokens", "--max_new_tokens", type=int,
                        required=False, default=2048)
    parser.add_argument("--num-ctx", "--num_ctx", type=int, required=False,
                        default=2048, help="ollama num_ctx")
    parser.add_argument("--out", type=str, required=False,
                        default=DEFAULT_OUT_PATH, help="verified JSONL 输出路径")
    parser.add_argument("--tokenizer-file", "--tokenizer_file", type=str,
                        required=False, default=None,
                        help="统计字段长度用的 tokenizer.json（可选）")
    parser.add_argument("--seed", type=int, required=False, default=42)
    return parser.parse_args()


# 测试代码 main 函数
def main():
    from utils.random_seed import set_seed_torch

    args = args_parse()
    set_seed_torch(args.seed)

    teacher_model = None
    tokenizer = None
    if args.backend == "local":
        from inference.loading import load_model

        teacher_model, _, extras = load_model(args.model)
        tokenizer = build_tokenizer(args.model, extras)
    elif args.tokenizer_file:
        from layers.tokenizers.tokenization import choose_tokenizer

        tokenizer = choose_tokenizer("qwen3", tokenizer_file_path=args.tokenizer_file)

    records = generate_teacher_dataset(
        backend=args.backend, model_name=args.model,
        num_samples=args.num_samples, max_new_tokens=args.max_new_tokens,
        out_path=args.out, teacher_model=teacher_model, tokenizer=tokenizer,
        seed=args.seed, num_ctx=args.num_ctx, train_size=args.train_size,
    )
    if tokenizer is not None:
        for row in average_field_lengths(args.out, tokenizer):
            logger.info(
                f"{row['field']}: avg={row['avg_tokens']:.2f} "
                f"min={row['min_tokens']} max={row['max_tokens']} n={row['count']}"
            )
    logger.info(f"Done. records={len(records)} out={args.out}")

if __name__ == "__main__":
    main()
