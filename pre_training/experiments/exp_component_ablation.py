# -*- coding: utf-8 -*-

# ***************************************************
# * File        : exp_component_ablation.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2026-10-28
# * Version     : 1.0.102801
# * Description : 组件升级消融实验（standalone，不进 CI）：同一 tiny Transformer
# *               骨架，逐项把 GPT-2 组件替换为现代化组件（LayerNorm->RMSNorm、
# *               GELU FFN->SwiGLU、learned PE->RoPE、MHA->GQA），同预算训练，
# *               对比 next-token loss 曲线。组件全部复用 layers/ 现有实现。
# *               设计对照 NamiGPT 进阶篇（GPT-2 -> LLaMA 的组件替换教学）。
# * Link        : https://github.com/aa1143/NamiGPT
# * Requirement : torch；CPU 分钟级（默认配置）
# ***************************************************

# python libraries
import argparse
import csv
import sys
import time
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)

import torch
import torch.nn as nn

from layers.attention import (
    GroupedQueryAttention,
    MultiHeadAttention,
    MultiHeadAttentionRoPE,
)
from layers.feed_forward import FeedForwardGELU, FeedForwardSiLU
from layers.normalization.layer_norm import LayerNorm
from layers.normalization.rms_norm import RMSNorm
from utils.log_util import logger

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]

# 消融配置：在 baseline 上逐项替换；gqa 依赖 RoPE（仓库 GQA 实现内置 RoPE），
# llama_style 为全量替换（RMSNorm + SwiGLU + GQA/RoPE，即 Llama 系组合）
VARIANTS = {
    "baseline":   dict(norm="layer", ffn="gelu",   attn="mha",  pos="learned"),
    "+rmsnorm":   dict(norm="rms",   ffn="gelu",   attn="mha",  pos="learned"),
    "+swiglu":    dict(norm="layer", ffn="swiglu", attn="mha",  pos="learned"),
    "+rope":      dict(norm="layer", ffn="gelu",   attn="mha_rope", pos="rope"),
    "+gqa":       dict(norm="layer", ffn="gelu",   attn="gqa",  pos="rope"),
    "llama_style": dict(norm="rms",  ffn="swiglu", attn="gqa",  pos="rope"),
}


class ConfigurableBlock(nn.Module):
    """pre-LN block，按 variant 组合 layers/ 现有组件（无 dropout）。"""

    def __init__(self, cfg, variant):
        super().__init__()
        d = cfg.embed_dim
        norm_cls = LayerNorm if variant["norm"] == "layer" else RMSNorm
        self.norm1, self.norm2 = norm_cls(d), norm_cls(d)
        if variant["attn"] == "mha":
            self.attn = MultiHeadAttention(
                d_model=d, d_out=d, n_heads=cfg.n_heads,
                context_length=cfg.context_length, dropout=0.0, qkv_bias=False,
            )
        elif variant["attn"] == "mha_rope":
            self.attn = MultiHeadAttentionRoPE(
                d_model=d, d_out=d, n_heads=cfg.n_heads,
                context_length=cfg.context_length, dtype=torch.float32,
            )
        elif variant["attn"] == "gqa":
            self.attn = GroupedQueryAttention(
                d_model=d, d_out=d, n_heads=cfg.n_heads,
                context_length=cfg.context_length,
                num_kv_groups=cfg.n_kv_groups, dtype=torch.float32,
            )
        else:
            raise ValueError(f"unknown attn variant: {variant['attn']}")
        self.ff = FeedForwardGELU(cfg) if variant["ffn"] == "gelu" else FeedForwardSiLU(cfg)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.ff(self.norm2(x))


class AblationGPT(nn.Module):
    """可配置组件的 tiny GPT：variant 决定 norm/ffn/attn/pos 的组合。"""

    def __init__(self, cfg, variant):
        super().__init__()
        self.cfg = cfg
        self.variant = variant
        self.tok_embed = nn.Embedding(cfg.vocab_size, cfg.embed_dim)
        self.use_pos_embed = variant["pos"] == "learned"
        if self.use_pos_embed:
            self.pos_embed = nn.Embedding(cfg.context_length, cfg.embed_dim)
        self.blocks = nn.ModuleList(
            ConfigurableBlock(cfg, variant) for _ in range(cfg.n_layers)
        )
        norm_cls = LayerNorm if variant["norm"] == "layer" else RMSNorm
        self.final_norm = norm_cls(cfg.embed_dim)
        self.out_head = nn.Linear(cfg.embed_dim, cfg.vocab_size, bias=False)

    def forward(self, ids):
        x = self.tok_embed(ids)
        if self.use_pos_embed:
            pos = torch.arange(ids.size(1), device=ids.device)
            x = x + self.pos_embed(pos).unsqueeze(0)
        for block in self.blocks:
            x = block(x)
        return self.out_head(self.final_norm(x))


def make_induction_data(batch_size, seq_len, vocab_size, device, generator):
    """合成语料：随机段 A + 分隔符 + 重复段 A（归纳头任务）。

    对重复段做 next-token 预测时，注意力必须回看第一段中同一 token 的
    **后继**才能压低 loss——这正是 induction head 机制，2 层 transformer
    数百步即可学会；loss 只在重复段上计算，理论下界为 0。
    """
    half = (seq_len - 1) // 2
    # A 使用 0..V-2，保留 V-1 作为分隔符
    A = torch.randint(0, vocab_size - 1, (batch_size, half), generator=generator).to(device)
    sep = torch.full((batch_size, 1), vocab_size - 1, dtype=torch.long, device=device)
    return torch.cat([A, sep, A], dim=1)


def train_one_variant(name, variant, args, device):
    torch.manual_seed(args.seed)
    generator = torch.Generator().manual_seed(args.seed + 1)

    cfg = argparse.Namespace(
        vocab_size=args.vocab_size, context_length=args.seq_len,
        embed_dim=args.embed_dim, d_ff=args.embed_dim * 4,
        n_heads=args.n_heads, n_kv_groups=max(1, args.n_heads // 2),
        n_layers=args.n_layers, dropout=0.0, dtype=torch.float32,
    )
    if variant["ffn"] == "swiglu":
        # SwiGLU 中间维取 ~2/3 * 4d，贴近 Llama 系约定（无偏置，参数量与 GELU 版接近）
        cfg.d_ff = cfg.embed_dim * 11 // 6

    model = AblationGPT(cfg, variant).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)

    curve = []
    t0 = time.time()
    half = (args.seq_len - 1) // 2
    for step in range(1, args.steps + 1):
        ids = make_induction_data(args.batch_size, args.seq_len, args.vocab_size, device, generator)
        logits = model(ids[:, :-1])
        targets = ids[:, 1:]
        # 只在重复段上计损失（输入位置 >= half 即目标位于第二段 A）
        mask = torch.arange(ids.size(1) - 1, device=device) >= half
        loss = nn.functional.cross_entropy(
            logits[:, mask].reshape(-1, args.vocab_size),
            targets[:, mask].reshape(-1),
        )
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % args.log_every == 0 or step == args.steps:
            curve.append((step, float(loss)))
            logger.info(f"[{name}] step {step:4d}/{args.steps}  loss {float(loss):.4f}")
    return dict(name=name, params=n_params, curve=curve,
                final=curve[-1][1], seconds=time.time() - t0)


def main():
    parser = argparse.ArgumentParser(description="组件升级消融实验（tiny，CPU 可跑）")
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--vocab-size", type=int, default=64)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--out-dir", type=str, default="results/ablation")
    args = parser.parse_args()

    device = torch.device("cpu")
    results = []
    for name, variant in VARIANTS.items():
        logger.info(f"{'='*20} {name}: {variant} {'='*20}")
        results.append(train_one_variant(name, variant, args, device))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "component_ablation.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["variant", "params", "final_loss", "seconds"])
        for r in results:
            writer.writerow([r["name"], r["params"], f"{r['final']:.4f}", f"{r['seconds']:.1f}"])
    with open(out_dir / "loss_curves.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["variant", "step", "loss"])
        for r in results:
            for step, loss in r["curve"]:
                writer.writerow([r["name"], step, f"{loss:.4f}"])

    logger.info(f"{'='*20} 汇总（{args.steps} 步，seed={args.seed}） {'='*20}")
    logger.info(f"{'variant':<12} {'params':>10} {'final_loss':>10} {'sec':>6}")
    for r in sorted(results, key=lambda x: x["final"]):
        logger.info(f"{r['name']:<12} {r['params']:>10,} {r['final']:>10.4f} {r['seconds']:>6.1f}")
    logger.info(f"结果已写入 {out_dir}/component_ablation.csv 与 loss_curves.csv")


if __name__ == "__main__":
    main()
