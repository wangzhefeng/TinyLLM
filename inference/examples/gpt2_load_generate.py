# -*- coding: utf-8 -*-

# ***************************************************
# * File        : gpt2_load_generate.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2026-10-28
# * Version     : 1.0.102801
# * Description : 生成章节端到端示例：加载官方 GPT-2 124M -> 贪心/采样生成 ->
# *               interpretability.trace_generation 展示每步 top-k 候选概率。
# *               学习路线图阶段 2/3 的衔接示例。
# * Link        : link
# * Requirement : 本地缓存 downloaded_models/gpt2_model（缺失时给出下载提示后退出）
# ***************************************************

# python libraries
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)

import torch

from inference.generate_core import generate
from inference.interpretability import render_trace_md, trace_generation
from layers.tokenizers.tokenization import (
    choose_tokenizer,
    text_to_token_ids,
    token_ids_to_text,
)
from utils.log_util import logger

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]

GPT2_CACHE = Path("downloaded_models/gpt2_model/models--openai-community--gpt2")


def main():
    if not GPT2_CACHE.exists():
        logger.info(f"未找到本地 GPT-2 缓存 {GPT2_CACHE}。")
        logger.info("下载方式：from transformers import GPT2LMHeadModel; "
                    "GPT2LMHeadModel.from_pretrained('gpt2', cache_dir='./downloaded/gpt2_model')")
        sys.exit(0)

    from inference.loading import load_model

    model, config, extras = load_model("gpt2-124m", device="cpu")
    model.eval()
    tokenizer = choose_tokenizer("tiktoken_gpt2_bpe")
    prompt = "Every effort moves"
    ids = text_to_token_ids(prompt, tokenizer_model="tiktoken_gpt2_bpe")
    logger.info(f"prompt token ids: {ids.tolist()}")

    # 1) 贪心（temperature=0 -> argmax）
    torch.manual_seed(123)
    greedy_ids = generate(
        model, ids.clone(), max_new_tokens=12, context_length=None,
        temperature=0.0, eos_token_id=50256,
    )
    logger.info(f"贪心: {token_ids_to_text(greedy_ids, tokenizer_model='tiktoken_gpt2_bpe')!r}")

    # 2) 温度 + top-k 采样
    torch.manual_seed(123)
    sampled_ids = generate(
        model, ids.clone(), max_new_tokens=12, context_length=None,
        temperature=0.8, top_k=50, eos_token_id=50256,
    )
    logger.info(f"temperature=0.8, top_k=50: "
                f"{token_ids_to_text(sampled_ids, tokenizer_model='tiktoken_gpt2_bpe')!r}")

    # 3) 教学模式：每步 top-k 候选概率与熵
    torch.manual_seed(123)
    records, out_ids = trace_generation(
        model, ids.clone(), max_new_tokens=8, temperature=0.8, top_k=50,
        eos_token_id=50256, k=5, tokenizer=tokenizer,
    )
    logger.info(f"trace 输出:\n{render_trace_md(records)}")


if __name__ == "__main__":
    main()
