# -*- coding: utf-8 -*-

# ***************************************************
# * File        : interpretability.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2026-10-28
# * Version     : 1.0.102801
# * Description : 教学用可解释性工具：逐步生成的 top-k 概率/熵追踪，
# *               与 GPT-2 各层 attention 权重的 opt-in 捕获及 markdown 渲染。
# *               设计对照 NamiGPT ep15（interpretability）与网页演示
# *               （每步候选词概率），见 docs/30-inference/interpretability.md。
# * Link        : https://github.com/aa1143/NamiGPT
# * Requirement : torch
# ***************************************************

__all__ = [
    "trace_generation",
    "attention_weights",
    "render_trace_md",
    "render_attention_md",
]

# python libraries
from pathlib import Path

import torch

from layers.sampling import sample_next_token

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


@torch.inference_mode()
def trace_generation(model, token_ids: torch.Tensor, max_new_tokens: int,
                     context_length: int = None, temperature: float = 0.0,
                     top_k: int = None, top_p: float = None, k: int = 5,
                     eos_token_id: int = None, tokenizer=None):
    """逐步生成追踪（batch=1，无 KV cache 的教学路径）。

    循环与采样逻辑镜像 ``inference.generate_core.generate(use_cache=False)``：
    贪心（temperature<=0）时 ``out_ids`` 与其逐 token 一致；``eos_token_id``
    语义相同——完成步的 eos 不追加进序列，但会在该步记录 ``eos_stop=True``。

    每步记录一个 dict：

    - ``step`` / ``token_id`` / ``token``：步序号与采样出的 token
      （``tokenizer`` 为 None 时 ``token`` 为 None）；
    - ``entropy``：该步全词表分布的熵（nats）；
    - ``topk``：前 ``k`` 个候选 ``[(token_id, token, prob), ...]``，按概率降序；
    - ``eos_stop``：本步是否因采到 eos 结束。

    Returns:
        (records, out_ids)：records 为逐步 dict 列表，out_ids 为含 prompt 的完整序列。
    """
    from inference.generate_core import _resolve_ctx_len

    if token_ids.size(0) != 1:
        raise ValueError("trace_generation 目前只支持 batch=1（教学用途）")
    model.eval()
    ctx_len = _resolve_ctx_len(model, context_length)
    out_ids = token_ids.clone()
    records = []
    for step in range(max_new_tokens):
        logits = model(out_ids[:, -ctx_len:])
        step_logits = logits[:, -1, :].float()
        probs = torch.softmax(step_logits, dim=-1)
        log_probs = torch.log_softmax(step_logits, dim=-1)
        entropy = float(-(probs * log_probs).sum(dim=-1).mean())
        n_top = min(k, probs.size(-1))
        topk_probs, topk_ids = torch.topk(probs, n_top, dim=-1)
        next_idx = sample_next_token(step_logits, temperature=temperature, top_k=top_k, top_p=top_p)

        def _tok(tid):
            return tokenizer.decode([tid]) if tokenizer is not None else None

        rec = {
            "step": step,
            "token_id": int(next_idx[0, 0]),
            "token": _tok(int(next_idx[0, 0])),
            "entropy": entropy,
            "topk": [
                (int(topk_ids[0, i]), _tok(int(topk_ids[0, i])), float(topk_probs[0, i]))
                for i in range(n_top)
            ],
        }
        records.append(rec)
        if eos_token_id is not None and rec["token_id"] == eos_token_id:
            rec["eos_stop"] = True
            break
        rec["eos_stop"] = False
        out_ids = torch.cat([out_ids, next_idx], dim=1)
    return records, out_ids


@torch.inference_mode()
def attention_weights(model, token_ids: torch.Tensor):
    """捕获 GPT-2 各层 attention 权重（无 cache 全序列前向）。

    依赖 ``Model.forward(output_attentions=True)`` 的 opt-in 契约
    （当前实现于 ``models/gpt2/gpt2_124M.py``；与 KV cache 互斥）。

    Returns:
        (logits, attns)：attns 为每层一个 ``(batch, n_heads, T, T)`` 张量的列表，
        权重是 softmax 之后、dropout 之前的取值。
    """
    model.eval()
    logits, attns = model(token_ids, output_attentions=True)
    return logits, attns


def render_trace_md(records) -> str:
    """把 trace_generation 的 records 渲染成 markdown 表。"""
    lines = [
        "| step | token | token_id | entropy | top-k (token_id: prob) |",
        "|---|---|---|---|---|",
    ]
    for r in records:
        topk = ", ".join(f"{tid}: {p:.3f}" for tid, _, p in r["topk"])
        tok = (r["token"] if r["token"] is not None else "").replace("|", "\\|") or "-"
        lines.append(f"| {r['step']} | {tok} | {r['token_id']} | {r['entropy']:.3f} | {topk} |")
    return "\n".join(lines)


def render_attention_md(attn: torch.Tensor, tokens=None) -> str:
    """把单层 attention 权重（heads 平均）渲染成 markdown 表。

    Args:
        attn: ``(batch, n_heads, T, T)``；取 batch/head 平均得到 ``T x T``。
        tokens: 长度 T 的标签列表（如 token 字符串），None 时用下标。
    """
    m = attn.float().mean(dim=(0, 1)).cpu()
    T = m.size(0)
    labels = [str(t) for t in tokens] if tokens is not None else [str(i) for i in range(T)]
    labels = [lab.replace("|", "\\|") for lab in labels]
    lines = ["| from \\ to | " + " | ".join(labels) + " |",
             "|---" * (T + 1) + "|"]
    for i in range(T):
        row = " | ".join(f"{m[i, j]:.3f}" for j in range(T))
        lines.append(f"| {labels[i]} | {row} |")
    return "\n".join(lines)


# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
