# -*- coding: utf-8 -*-

# ***************************************************
# * File        : run.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2026-10-24
# * Version     : 1.0.102421
# * Description : RLVR 真实训练入口（GRPO / REINFORCE，reasoning ch06/ch07）。
# *               load_model 载入模型+tokenizer -> MATH-500 训练子集 ->
# *               verifier reward -> 逐 prompt 更新 -> 指标 CSV + checkpoint。
# *               重依赖（模型权重、数据集、matplotlib）全部懒加载。
# * Link        : https://github.com/rasbt/reasoning-from-scratch/tree/main/ch07/03_rlvr_grpo_scripts_advanced
# * Requirement : torch；按模型需要 transformers/tokenizers；数据需要 datasets
# ***************************************************

__all__ = [
    "args_parse",
    "run",
]

# python libraries
import argparse
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)

from utils.log_util import logger

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


def _task_from_argv(default="grpo"):
    """从 post_training/run.py 注入的 argv[0] 识别任务名（grpo / reinforce）。"""
    argv0 = sys.argv[0] if sys.argv else ""
    for algo in ("grpo", "reinforce"):
        if f"--task {algo}" in argv0:
            return algo
    return default


def args_parse():
    # ------------------------------
    # parser
    # ------------------------------
    parser = argparse.ArgumentParser(description="TinyLLM RLVR Training (GRPO / REINFORCE)")
    # ------------------------------
    # add arguments
    # ------------------------------
    # task params
    parser.add_argument("--algo", type=str, required=False, default=_task_from_argv(),
                        choices=["grpo", "reinforce"],
                        help="training algorithm")
    parser.add_argument("--model", type=str, required=False, default="qwen3-0.6b",
                        help="model key in MODEL_REGISTRY")
    # data params
    parser.add_argument("--train-size", "--train_size", type=int, required=False,
                        default=50,
                        help="number of MATH-500 training prompts (seed sampled)")
    parser.add_argument("--epochs", type=int, required=False, default=1,
                        help="number of epochs over the training subset")
    # rl params
    parser.add_argument("--group-size", "--group_size", type=int, required=False,
                        default=8,
                        help="rollouts per prompt (GRPO group / REINFORCE samples)")
    parser.add_argument("--max-new-tokens", "--max_new_tokens", type=int,
                        required=False, default=512,
                        help="max generated tokens per rollout")
    parser.add_argument("--temperature", type=float, required=False, default=0.8,
                        help="sampling temperature")
    # optimization params
    parser.add_argument("--lr", type=float, required=False, default=1e-5,
                        help="learning rate")
    parser.add_argument("--clip-eps", "--clip_eps", type=float, required=False,
                        default=0.2, help="PPO clip epsilon (lower bound)")
    parser.add_argument("--clip-eps-high", "--clip_eps_high", type=float,
                        required=False, default=None,
                        help="clip-higher upper epsilon (None = symmetric)")
    parser.add_argument("--kl-beta", "--kl_beta", type=float, required=False,
                        default=0.0, help="KL penalty coefficient (needs ref model)")
    parser.add_argument("--kl-estimator", "--kl_estimator", type=str, required=False,
                        default="k3", choices=["k1", "k3"], help="KL estimator")
    parser.add_argument("--loss-norm", "--loss_norm", type=str, required=False,
                        default="token_mean", choices=["token_mean", "seq_sum_mean"],
                        help="loss normalization (Dr.GRPO: seq_sum_mean)")
    parser.add_argument("--use-batched", "--use_batched", type=int, required=False,
                        default=0,
                        help="use batched group sampling (post_training/rl/batched.py)")
    # output params
    parser.add_argument("--save-every", "--save_every", type=int, required=False,
                        default=10, help="save checkpoint every N steps")
    parser.add_argument("--out", type=str, required=False,
                        default="results/checkpoints/grpo",
                        help="checkpoint output dir")
    parser.add_argument("--logs", type=str, required=False, default="logs/grpo",
                        help="metrics log dir (metrics.csv)")
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


def _build_tokenizer(model_key, extras):
    """按模型族构造 tokenizer（qwen3/gemma3/phi 需要 tokenizer_file_path）。"""
    from inference.registry import get_model_spec
    from layers.tokenizers.tokenization import choose_tokenizer

    family = get_model_spec(model_key).family
    if "tokenizer_file_path" in extras:
        return choose_tokenizer(family, tokenizer_file_path=extras["tokenizer_file_path"])
    return choose_tokenizer({
        "gpt2": "tiktoken_gpt2_bpe",
        "llama": "llama2",
    }.get(family, family))


def run(args):
    # ------------------------------
    # python libraries（重依赖懒加载）
    # ------------------------------
    import copy
    import random

    import torch

    from inference.evaluation.benchmarks.math_verifiers import (
        load_math500_data,
        render_prompt,
    )
    from inference.loading import load_model
    from inference.registry import get_model_spec
    from post_training.rl.batched import grpo_update_step_batched
    from post_training.rl.core import grpo_update_epoch
    from post_training.rl.rewards import correct_answer_reward
    from post_training.rl.tracking import MetricsTracker
    from training.checkpoint import save_training_checkpoint

    # ------------------------------
    # model & tokenizer
    # ------------------------------
    model, config, extras = load_model(args.model, device=args.device)
    tokenizer = _build_tokenizer(args.model, extras)
    spec = get_model_spec(args.model)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is None:
        eos_token_id = getattr(spec, "eos_token_id", None)
    context_length = getattr(config, "context_length", None)

    ref_model = None
    if args.kl_beta > 0.0:
        ref_model = copy.deepcopy(model)
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad = False
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # ------------------------------
    # data & rewards
    # ------------------------------
    math_data = load_math500_data(verbose=False)
    rng = random.Random(args.seed)
    train_subset = rng.sample(math_data, args.train_size) \
        if args.train_size < len(math_data) else list(math_data)

    examples = []
    base_reward = correct_answer_reward()
    for ex in train_subset:
        prompt = render_prompt(ex["problem"])
        prompt_ids = torch.tensor(
            tokenizer.encode(prompt), dtype=torch.long, device=args.device,
        ).unsqueeze(0)
        expected = ex["answer"]

        def decode_fn(row, prompt_len=prompt_ids.shape[1]):
            return tokenizer.decode(row[prompt_len:].tolist())

        def reward_fn(text, expected=expected):
            return base_reward(text, expected)

        examples.append((prompt_ids, decode_fn, reward_fn))

    # ------------------------------
    # training loop（逐 prompt 一步更新，reasoning ch07 工作流）
    # ------------------------------
    tracker = MetricsTracker()
    step = 0
    update_kwargs = dict(
        group_size=args.group_size, max_new_tokens=args.max_new_tokens,
        context_length=context_length, temperature=args.temperature,
        eos_token_id=eos_token_id,
    )
    if args.algo == "reinforce":
        from post_training.reasoning.training.reinforce import reinforce_step
    for epoch in range(1, args.epochs + 1):
        for prompt_ids, decode_fn, reward_fn in examples:
            step += 1
            if args.algo == "reinforce":
                res = reinforce_step(
                    model, optimizer, prompt_ids, reward_fn,
                    num_samples=args.group_size, decode_fn=decode_fn,
                    **{k: v for k, v in update_kwargs.items()
                       if k != "group_size"},
                )
                res = {"loss": res["loss"], "mean_reward": res["mean_reward"]}
            elif args.use_batched:
                res = grpo_update_step_batched(
                    model, optimizer, prompt_ids, reward_fn,
                    decode_fn=decode_fn, clip_eps=args.clip_eps,
                    clip_eps_high=args.clip_eps_high, kl_beta=args.kl_beta,
                    kl_estimator=args.kl_estimator, loss_norm=args.loss_norm,
                    ref_model=ref_model, **update_kwargs,
                )
                res = {"loss": res["loss"], "mean_reward": res["mean_reward"]}
            else:
                res = grpo_update_epoch(
                    model, optimizer, [prompt_ids], reward_fn,
                    decode_fn=decode_fn, clip_eps=args.clip_eps,
                    clip_eps_high=args.clip_eps_high, kl_beta=args.kl_beta,
                    kl_estimator=args.kl_estimator, loss_norm=args.loss_norm,
                    ref_model=ref_model, return_diagnostics=True,
                    **update_kwargs,
                )
            tracker.append(step=step, epoch=epoch, loss=res["loss"],
                           reward_mean=res["mean_reward"],
                           clip_frac=res.get("clip_frac", 0.0),
                           kl=res.get("kl", 0.0),
                           mean_completion_len=res.get("mean_completion_len", 0.0))
            logger.info(
                f"[{args.algo} step {step}] loss={res['loss']:.4f} "
                f"reward={res['mean_reward']:.3f}"
            )
            if step % args.save_every == 0:
                ckpt_path = Path(args.out) / f"{args.algo}_step{step:05d}.pt"
                save_training_checkpoint(
                    ckpt_path, model, optimizer=optimizer,
                    epoch=epoch, step=step,
                    extra={"algo": args.algo, "model": args.model},
                )
                logger.info(f"Saved checkpoint to {ckpt_path}")

    # ------------------------------
    # metrics
    # ------------------------------
    metrics_path = Path(args.logs) / f"{args.algo}_metrics.csv"
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
