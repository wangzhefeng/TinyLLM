# -*- coding: utf-8 -*-

# ***************************************************
# * File        : run.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2026-10-24
# * Version     : 1.0.102421
# * Description : ch08 蒸馏 SFT 训练入口：verified 教师数据 JSONL ->
# *               学生模型编码 (prompt, completion) ->
# *               distill_sft_epoch 批量训练 -> checkpoint + metrics CSV。
# *               教师数据先由 post_training/reasoning/data_gen.py 生成。
# *               重依赖（torch/模型权重）全部懒加载。
# * Link        : https://github.com/rasbt/reasoning-from-scratch/tree/main/ch08/04_train_with_distillation
# * Requirement : torch；按模型需要 transformers/tokenizers
# ***************************************************

__all__ = [
    "args_parse",
    "run",
]

# python libraries
import argparse
import json
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)

from utils.log_util import logger

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


def args_parse():
    # ------------------------------
    # parser
    # ------------------------------
    parser = argparse.ArgumentParser(description="TinyLLM Distillation SFT Training (reasoning ch08)")
    # ------------------------------
    # add arguments
    # ------------------------------
    # task params
    parser.add_argument("--model", type=str, required=False, default="qwen3-0.6b",
                        help="student model key in MODEL_REGISTRY")
    # data params
    parser.add_argument("--data", type=str, required=False,
                        default="dataset/finetuning/math_distill/train.jsonl",
                        help="teacher data JSONL from post_training.reasoning.data_gen")
    parser.add_argument("--epochs", type=int, required=False, default=2,
                        help="number of training epochs")
    parser.add_argument("--batch-size", "--batch_size", type=int, required=False,
                        default=4, help="examples per optimization step")
    # optimization params
    parser.add_argument("--lr", type=float, required=False, default=5e-5,
                        help="learning rate")
    parser.add_argument("--max-grad-norm", "--max_grad_norm", type=float,
                        required=False, default=1.0, help="gradient clipping norm")
    # output params
    parser.add_argument("--save-every", "--save_every", type=int, required=False,
                        default=1, help="save checkpoint every N epochs")
    parser.add_argument("--out", type=str, required=False,
                        default="results/checkpoints/distill",
                        help="checkpoint output dir")
    parser.add_argument("--logs", type=str, required=False, default="logs/distill",
                        help="metrics log dir (distill_metrics.csv)")
    # runtime params
    parser.add_argument("--seed", type=int, required=False, default=42,
                        help="random seed")
    parser.add_argument("--device", type=str, required=False, default="cpu",
                        help="device, e.g. cpu / mps / cuda")
    # ------------------------------
    # arguments parse
    # ------------------------------
    args = parser.parse_args()
    logger.info(f"Args in experiment: \n{args}")
    return args


def _load_dataset(args, tokenizer, context_length):
    """读取 verified JSONL 并编码为 (prompt_ids, completion_ids) 列表。"""
    import torch

    from inference.evaluation.benchmarks.math_verifiers import render_prompt

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(
            f"教师数据不存在: {data_path}。"
            "请先运行 python -m post_training.reasoning.data_gen 生成"
            "（--backend ollama/openrouter 需要对应服务与凭证）"
        )
    records = [
        json.loads(line) for line in
        data_path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]

    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    dataset = []
    skipped = 0
    for record in records:
        prompt = record.get("prompt") or render_prompt(record["problem"])
        prompt_ids = tokenizer.encode(prompt)
        completion_ids = tokenizer.encode(record["completion"])
        if eos_token_id is not None:
            completion_ids = completion_ids + [eos_token_id]
        if not completion_ids or (
            context_length is not None
            and len(prompt_ids) + len(completion_ids) > context_length
        ):
            skipped += 1
            continue
        dataset.append((
            torch.tensor(prompt_ids, dtype=torch.long, device=args.device).unsqueeze(0),
            torch.tensor(completion_ids, dtype=torch.long, device=args.device).unsqueeze(0),
        ))
    logger.info(
        f"Loaded {len(dataset)} examples from {data_path} (skipped {skipped})"
    )
    return dataset


def run(args):
    # ------------------------------
    # python libraries（重依赖懒加载）
    # ------------------------------
    import random

    import torch

    from inference.loading import load_model
    from post_training.reasoning.data_gen import build_tokenizer
    from post_training.reasoning.training.distill_reasoning import distill_sft_epoch
    from post_training.rl.tracking import MetricsTracker
    from training.checkpoint import save_training_checkpoint

    # ------------------------------
    # model & tokenizer
    # ------------------------------
    model, config, extras = load_model(args.model, device=args.device)
    tokenizer = build_tokenizer(args.model, extras)
    context_length = getattr(config, "context_length", None)
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = getattr(tokenizer, "eos_token_id", None)
    if pad_id is None:
        pad_id = 0

    # ------------------------------
    # data
    # ------------------------------
    dataset = _load_dataset(args, tokenizer, context_length)
    if not dataset:
        raise RuntimeError("有效训练样本为 0：检查 JSONL completion 与 context_length")
    rng = random.Random(args.seed)
    rng.shuffle(dataset)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # ------------------------------
    # training loop（批量蒸馏 SFT，reasoning ch08 distill_batched 工作流）
    # ------------------------------
    tracker = MetricsTracker()
    for epoch in range(1, args.epochs + 1):
        loss = distill_sft_epoch(
            dataset, model, optimizer, batch_size=args.batch_size,
            pad_id=pad_id, max_grad_norm=args.max_grad_norm,
        )
        tracker.append(epoch=epoch, loss=loss)
        logger.info(f"[distill epoch {epoch}/{args.epochs}] loss={loss:.4f}")
        if epoch % args.save_every == 0:
            ckpt_path = Path(args.out) / f"distill_epoch{epoch:03d}.pt"
            save_training_checkpoint(
                ckpt_path, model, optimizer=optimizer,
                epoch=epoch, step=epoch,
                extra={"algo": "distill", "model": args.model},
            )
            logger.info(f"Saved checkpoint to {ckpt_path}")

    # ------------------------------
    # metrics
    # ------------------------------
    metrics_path = Path(args.logs) / "distill_metrics.csv"
    tracker.to_csv(metrics_path)
    logger.info(f"Metrics saved to {metrics_path}")
    for col, stats in tracker.summary().items():
        logger.info(f"{col}: mean={stats['mean']:.4f} last={stats['last']:.4f}")


# 测试代码 main 函数
def main():
    args = args_parse()
    from utils.random_seed import set_seed_torch
    set_seed_torch(args.seed)
    run(args)

if __name__ == "__main__":
    main()
