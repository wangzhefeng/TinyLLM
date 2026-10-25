# -*- coding: utf-8 -*-

# ***************************************************
# * File        : benchmark_compile.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2026-10-24
# * Version     : 1.0.102421
# * Description : eager vs torch.compile 生成吞吐（tokens/s）对比脚本。
# *               设备感知：仅 CUDA 后端做 compile 对比（Inductor 加速
# *               成熟）；MPS/CPU 打印说明后只跑 eager。
# *               统一走 inference.loading.load_model + generate_stats，
# *               每个 repeat 报告 Time / tokens/sec，最后汇总平均值与加速比。
# * Link        : https://pytorch.org/tutorials/recipes/torch_compile_caching.html
# * Requirement : torch；模型权重（qwen3 首次运行需联网下载）
# ***************************************************

__all__ = [
    "parse_args",
    "benchmark_backend",
    "main",
]

# python libraries
import argparse
import os
import sys
import time
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]
os.environ['LOG_NAME'] = LOGGING_LABEL
from utils.log_util import logger


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="eager vs torch.compile 生成吞吐对比（tokens/s）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", default="qwen3-0.6b",
                        help="MODEL_REGISTRY 键")
    parser.add_argument("--max-new-tokens", type=int, default=128,
                        help="每次生成的最大新 token 数")
    parser.add_argument("--repeat", type=int, default=3,
                        help="每个后端计时的运行次数（另含 1 次预热不计入）")
    return parser.parse_args(argv)


def _pick_generator(family):
    from inference.generate_core import generate, generate_qwen3

    return generate_qwen3 if family == "qwen3" else generate


def _generated_tokens(out, prompt_len, family):
    # generate_qwen3 只返回新增部分；generate 返回含 prompt 的完整序列
    return out if family == "qwen3" else out[:, prompt_len:]


def benchmark_backend(backend_model, gen, token_ids, max_new_tokens,
                      ctx_len, eos_token_id, family, repeat, label):
    """跑 1 次预热 + repeat 次计时，返回每轮 (tokens, sec) 列表。

    每轮用 generate_stats 报告 Time / tokens/sec（CUDA 计时前后同步）。
    """
    from inference.generate_core import generate_stats

    device_type = token_ids.device.type
    runs = []
    for i in range(repeat + 1):
        prompt_len = token_ids.size(1)
        if device_type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        out = gen(
            backend_model, token_ids, max_new_tokens=max_new_tokens,
            context_length=ctx_len, temperature=0.0,
            eos_token_id=eos_token_id, use_cache=True,
        )
        if device_type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        new_tokens = _generated_tokens(out, prompt_len, family)
        generate_stats(new_tokens, None, start, elapsed)
        if i == 0:
            logger.info(f"[{label}] warmup run（编译/缓存开销不计入统计）")
        else:
            runs.append((int(new_tokens.numel()), elapsed))
    return runs


def _summarize(label, runs):
    total_tokens = sum(t for t, _ in runs)
    total_time = sum(s for _, s in runs)
    tokens_per_sec = total_tokens / total_time if total_time else 0.0
    logger.info(f"[{label}] avg {tokens_per_sec:.0f} tokens/s "
                f"({total_tokens} tokens in {total_time:.2f}s, {len(runs)} runs)")
    return tokens_per_sec


def main(argv=None):
    args = parse_args(argv)

    import torch
    from inference.generate_core import _resolve_ctx_len
    from inference.loading import load_model
    from models.registry import get_model_spec
    from layers.tokenizers.tokenization import choose_tokenizer

    if torch.cuda.is_available():
        device, device_type = torch.device("cuda"), "cuda"
    elif torch.backends.mps.is_available():
        device, device_type = torch.device("mps"), "mps"
    else:
        device, device_type = torch.device("cpu"), "cpu"
    logger.info(f"device: {device} ({device_type})")

    spec = get_model_spec(args.model)
    model, config, extras = load_model(args.model, device=str(device))
    name = spec.tokenizer_model or {
        "gpt2": "tiktoken_gpt2_bpe", "llama": "llama2",
    }.get(spec.family, spec.family)
    tokenizer_kwargs = {}
    if extras.get("tokenizer_file_path"):
        tokenizer_kwargs["tokenizer_file_path"] = extras["tokenizer_file_path"]
    tokenizer = choose_tokenizer(name, **tokenizer_kwargs)

    prompt = "Explain what a large language model is in one paragraph."
    token_ids = torch.tensor(
        tokenizer.encode(prompt), dtype=torch.long, device=device
    ).unsqueeze(0)
    gen = _pick_generator(spec.family)
    ctx_len = _resolve_ctx_len(model, None)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)

    logger.info(f"model={args.model} max_new_tokens={args.max_new_tokens} "
                f"repeat={args.repeat}")

    eager_runs = benchmark_backend(
        model, gen, token_ids.clone(), args.max_new_tokens,
        ctx_len, eos_token_id, spec.family, args.repeat, "eager",
    )
    eager_tps = _summarize("eager", eager_runs)

    if device_type != "cuda":
        logger.info(
            f"当前设备 {device_type}：torch.compile 的 Inductor 后端仅在 CUDA 上"
            "有成熟加速收益，MPS/CPU 跳过 compile 对比，只报告 eager 结果。"
        )
        return

    compiled = torch.compile(model)
    compiled_runs = benchmark_backend(
        compiled, gen, token_ids.clone(), args.max_new_tokens,
        ctx_len, eos_token_id, spec.family, args.repeat, "compiled",
    )
    compiled_tps = _summarize("compiled", compiled_runs)

    speedup = compiled_tps / eager_tps if eager_tps else 0.0
    logger.info(f"speedup (compiled / eager): {speedup:.2f}x")


# 测试代码 main 函数
if __name__ == "__main__":
    main()
