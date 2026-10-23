# -*- coding: utf-8 -*-

# ***************************************************
# * File        : main.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2026-10-24
# * Version     : 1.0.102409
# * Description : 06 单元：流水线并行（Pipeline Parallelism, PP）。使用
# *               PyTorch 原生 torch.distributed.pipelining API，把
# *               _common/model.py 的 toy Transformer 按层切成 2 个 stage
# *               （2 进程 = 2 stage），用 ScheduleGPipe / Schedule1F1B
# *               跑 next-token 交叉熵训练。CPU + gloo 可运行。
# *               --verify 模式做"单进程整 batch vs 2 进程流水线"的
# *               数值等价性对照。
# * Link        : https://pytorch.org/docs/stable/distributed.pipelining.html
# * Requirement : torch >= 2.5；gloo（CPU）或 nccl（CUDA）；torchrun 2 进程
# ***************************************************

__all__ = []

# python libraries
import argparse
import sys
import time
from pathlib import Path

# repo root: tools/distributed/06-pp/main.py -> parents[3]
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.pipelining import (
    PipelineStage,
    Schedule1F1B,
    ScheduleGPipe,
    SplitPoint,
    pipeline,
)

from tools.distributed._common import dist as dist_utils
from tools.distributed._common.metrics import log_step_metrics
from tools.distributed._common.model import ModelArgs, Transformer

# ----------------------------------------------------------------------------
# 教学点 1：什么是流水线并行（PP）
#
# 大模型放不进单卡显存时，可以把模型"纵向"按层切成若干段（stage），每张卡
# （本单元是每个进程）只放一段。前向时激活（activation）依次流过
# stage0 -> stage1 -> ... -> stageN；反向时梯度反方向流回。stage 之间的
# 张量传递（send/recv）由 torch.distributed.pipelining 自动完成，用户代码
# 里看不到通信细节。
#
# 本单元：n_layers=2 的 toy Transformer 切成 2 个 stage：
#   stage0（rank 0）：tok_embeddings + pos_embeddings + dropout + layers.0
#   stage1（rank 1）：layers.1 + norm + output（输出头）
# ----------------------------------------------------------------------------


def make_batch(step: int, seed: int, batch_size: int, seq_len: int, vocab_size: int):
    """
    造 next-token 样本：x = tokens[:, :-1]，y = tokens[:, 1:]。

    序列按"首 token 随机 + 固定转移规则 t' = (3t + 1) % vocab_size"生成：
    纯随机 token 的 next-token 交叉熵下界是 ln(vocab_size)，loss 降不下去，
    教学演示看不出训练效果；固定转移规则让任务可学习，loss 能降到接近 0。

    每个 step 用 (seed + step) 重设 RNG，保证：
    - 所有 rank 生成完全相同的 batch（流水线各 stage 需要同一份数据切片）；
    - --verify 模式下，单进程基线与流水线看到完全相同的数据序列。
    """
    g = torch.Generator().manual_seed(seed + step)
    first = torch.randint(0, vocab_size, (batch_size, 1), generator=g)
    tokens = [first]
    for _ in range(seq_len):
        tokens.append((tokens[-1] * 3 + 1) % vocab_size)
    tokens = torch.cat(tokens, dim=1)  # (batch_size, seq_len + 1)
    # contiguous()：torch 2.5.1 的 pipelining 会校验输入 stride 与追踪样例
    # 一致，切片产生的非连续 view（stride 16 vs 15）会报 PipeliningShapeError
    return tokens[:, :-1].contiguous(), tokens[:, 1:].contiguous()


# ----------------------------------------------------------------------------
# 教学点 2：microbatch 与"气泡"（bubble）
#
# 朴素流水线一次只跑一个 batch：stage0 算完前向交给 stage1，期间 stage0 空
# 闲；反向时同样只有一个 stage 在工作。设备空闲的时间比例称为"气泡"。
# 解决办法是把一个 batch 切成 n 个 microbatch 流水推进：stage1 还在算第 0
# 个 microbatch 时，stage0 已经开始算第 1 个。气泡占比约为
# (n_stages - 1) / (n_microbatches + n_stages - 1)，microbatch 越多气泡越小。
#
# 调度方式（schedule）决定 microbatch 的前向/反向如何交错：
# - GPipe：先做完所有 microbatch 的前向，再按相反顺序做所有反向。实现简单，
#   但每个 stage 要同时缓存全部 microbatch 的激活，显存峰值高；
# - 1F1B：稳态时每个 stage 交替做"1 次前向 + 1 次反向"，尽早释放激活，
#   显存占用显著降低，气泡占比与 GPipe 相同。
# ----------------------------------------------------------------------------


def build_stage(
    model_args: ModelArgs,
    seed: int,
    mb_size: int,
    rank: int,
    device: torch.device,
) -> tuple[PipelineStage, int]:
    """
    每个 rank 用相同 seed 构造同一份完整模型（参数初始化一致），再用
    pipeline() 按 split_spec 切成 2 个 stage，本 rank 只实例化自己的那段。

    split_spec 的 {"layers.0": SplitPoint.END} 表示：在 layers.0 之后切开，
    layers.0 及其之前的部分归 stage0，其余归 stage1。

    注意（torch 2.5.1 的坑）：pipeline() 会把输入样例的 shape 固化进
    Pipe 的输入元数据，运行时每个 microbatch 的 shape 必须与样例一致，
    否则报 PipeliningShapeError。因此样例必须是真实的 microbatch 形状
    (mb_size, seq_len)，而不是整 batch。
    """
    torch.manual_seed(seed)
    model = Transformer(model_args)

    # pipeline() 需要一个 microbatch 形状的真实输入样例（用于追踪/切分），
    # 用独立 generator 生成，不消耗全局 RNG
    g = torch.Generator().manual_seed(1234)
    mb_example = torch.randint(0, model_args.vocab_size, (mb_size, model_args.max_seq_len - 1), generator=g)

    pipe = pipeline(
        model,
        (mb_example,),
        split_spec={"layers.0": SplitPoint.END},
    )
    assert pipe.num_stages == 2, f"期望 2 个 stage，实际 {pipe.num_stages}"

    # 打印本 rank 持有的子模块与参数量：直观感受"模型被切开了"
    submod = pipe.get_stage_module(rank)
    n_params = sum(p.numel() for p in submod.parameters())
    top_modules = [name for name, _ in submod.named_children()]
    print(
        f"[rank {rank}] stage {rank}/{pipe.num_stages - 1} | "
        f"params {n_params} | top submodules: {top_modules}",
        flush=True,
    )

    # build_stage 把本 stage 的子模块搬到 device 并包装成 PipelineStage；
    # 之后激活在 stage 间的 send/recv 由 stage + schedule 协作完成
    stage = pipe.build_stage(rank, device)
    return stage, pipe.num_stages


def run_pipeline(
    args,
    stage: PipelineStage,
    model_args: ModelArgs,
    rank: int,
    world_size: int,
    device: torch.device,
) -> list[float]:
    """
    用指定的 schedule 跑 args.steps 步训练，返回每步的整 batch 平均 loss。

    关键语义（等价性对照的基础）：
    torch 2.5.1 的 ScheduleGPipe/Schedule1F1B 不会对 loss_fn 的结果做
    microbatch 平均——每个 microbatch 的 loss 直接 backward，梯度是各
    microbatch 梯度的累加。因此这里在 loss_fn 里手动除以 n_microbatches，
    使一次 optimizer.step() 的梯度与"整 batch 一次 backward"严格同语义；
    各 microbatch 的 loss 之和（microbatch 等大时）恰好等于整 batch 平均
    loss，用于日志与对照。
    """
    is_last_stage = rank == world_size - 1

    def loss_fn(logits, targets):
        loss = F.cross_entropy(
            logits.reshape(-1, model_args.vocab_size),
            targets.reshape(-1),
        )
        return loss / args.n_microbatches

    schedule_cls = ScheduleGPipe if args.schedule == "gpipe" else Schedule1F1B
    schedule = schedule_cls(stage, args.n_microbatches, loss_fn=loss_fn)

    # 每个 stage 只优化自己持有的参数（各 rank 一个 optimizer，互不重叠）
    optimizer = torch.optim.AdamW(stage.submod.parameters(), lr=args.lr)

    step_losses: list[float] = []
    for step in range(args.steps):
        t0 = time.perf_counter()
        x, y = make_batch(step, args.seed, args.batch_size, model_args.max_seq_len - 1, model_args.vocab_size)
        x = x.to(device)
        y = y.to(device)

        # 通信由框架代劳：非首 stage 的 step() 内部会 recv 上一个 stage 的
        # 激活，首 stage 算完自动 send；反向同理。用户只需：
        # - 第一个 stage：传入输入 x；
        # - 最后一个 stage：传入 target 和 losses 容器（loss 在这里计算）。
        if rank == 0:
            schedule.step(x)
        elif is_last_stage:
            losses = []
            schedule.step(target=y, losses=losses)
        else:  # 中间 stage（本单元 2 stage 不会走到）
            schedule.step()

        optimizer.step()
        optimizer.zero_grad()

        # 整 batch 平均 loss = 各 microbatch（已除 n_microbatches）loss 之和；
        # 从最后一个 stage 广播到 rank 0 统一打日志
        loss_val = sum(float(l) for l in losses) if is_last_stage else 0.0
        loss_t = torch.tensor([loss_val], device=device)
        dist.broadcast(loss_t, src=world_size - 1)
        step_losses.append(float(loss_t))

        if step % args.log_interval == 0:
            dt = time.perf_counter() - t0
            n_tokens = args.batch_size * (model_args.max_seq_len - 1)
            log_step_metrics(
                step=step,
                loss=float(loss_t),
                tokens_per_sec=n_tokens / dt if dt > 0 else None,
                extra=f"schedule {args.schedule} | microbatches {args.n_microbatches}",
            )
    return step_losses


def run_baseline(args, model_args: ModelArgs) -> list[float]:
    """
    等价性对照的单进程基线：同 seed、同数据、整 batch 一次 backward。

    与流水线的差异只在计算组织方式（microbatch 顺序累加 vs 整 batch），
    数学上同语义；浮点累加顺序不同会带来 ~1e-6 量级的数值差异。
    注意：verify 模式强制 dropout=0，否则 microbatch 拆分改变 dropout 的
    RNG 消耗顺序，两边随机掩码不同，无法严格对齐。
    """
    torch.manual_seed(args.seed)
    model = Transformer(model_args)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    step_losses: list[float] = []
    for step in range(args.steps):
        x, y = make_batch(step, args.seed, args.batch_size, model_args.max_seq_len - 1, model_args.vocab_size)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, model_args.vocab_size), y.reshape(-1))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        step_losses.append(float(loss))
        if step % args.log_interval == 0:
            log_step_metrics(step=step, loss=float(loss), extra="baseline (single process)")
    return step_losses


def main():
    parser = argparse.ArgumentParser(description="06-pp: torch.distributed.pipelining 流水线并行示例")
    parser.add_argument("--schedule", choices=["gpipe", "1f1b"], default="gpipe", help="流水线调度方式")
    parser.add_argument("--n-microbatches", type=int, default=4, help="每个 batch 切成的 microbatch 数")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16, help="整 batch 大小（会被切成 n-microbatches 份）")
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--verify", action="store_true", help="跑单进程基线 vs 流水线的数值等价性对照（强制 dropout=0）")
    args = parser.parse_args()

    if args.batch_size % args.n_microbatches != 0:
        raise ValueError("batch-size 必须能被 n-microbatches 整除（microbatch 等大，loss 求和语义才等于整 batch 平均）")

    # 实测结论（torch 2.5.1）：Schedule1F1B 在 CPU + gloo 后端下会触发 gloo
    # 传输层的 C++ 硬崩溃（gloo::EnforceNotMet: Cannot lock pointer to
    # unbound buffer，macOS libuv 传输），进程直接 abort、拿不到 Python 异常；
    # 最小 nn.Module 复现用例同样崩溃，属框架级 bug。ScheduleGPipe 不受影响。
    # 1F1B 请在 CUDA + NCCL 环境下使用，或升级 torch 版本后再试。
    if args.schedule == "1f1b" and not torch.cuda.is_available():
        raise RuntimeError(
            "torch 2.5.1 的 Schedule1F1B 在 CPU + gloo 后端下会触发 gloo 硬崩溃"
            "（Cannot lock pointer to unbound buffer），无法运行；"
            "请改用 --schedule gpipe，或在 CUDA + NCCL 环境下运行 1F1B。"
        )

    rank, world_size, _local_rank = dist_utils.setup()
    if world_size != 2:
        raise RuntimeError(f"本单元按 n_layers=2 切成 2 个 stage，需要 torchrun 2 进程，当前 world_size={world_size}")
    device = torch.device("cpu")  # CPU + gloo；有 CUDA 时 pipelining 同样可用（nccl）

    model_args = ModelArgs(dropout_p=0.0 if args.verify else args.dropout)
    dist_utils.rank0_print(
        f"config: schedule={args.schedule} microbatches={args.n_microbatches} "
        f"batch={args.batch_size} steps={args.steps} lr={args.lr} dropout={model_args.dropout_p}"
    )

    if args.verify:
        # 两个 rank 各自算一遍基线（同 seed 结果相同），避免额外通信
        dist_utils.rank0_print("=== baseline: 单进程整 batch 训练 ===")
        baseline_losses = run_baseline(args, model_args)

    stage, _num_stages = build_stage(model_args, args.seed, args.batch_size // args.n_microbatches, rank, device)

    dist_utils.rank0_print(f"=== pipeline: {args.schedule} 调度，2 stage 训练 ===")
    pipe_losses = run_pipeline(args, stage, model_args, rank, world_size, device)

    if args.verify:
        # 对照：同 seed 同数据下，两条路径的每步 loss 应几乎一致
        max_diff = max(abs(a - b) for a, b in zip(baseline_losses, pipe_losses))
        tol = 1e-3
        dist_utils.rank0_print(
            f"=== verify 结果 ===\n"
            f"baseline final loss: {baseline_losses[-1]:.6f}\n"
            f"pipeline final loss: {pipe_losses[-1]:.6f}\n"
            f"max per-step |diff|: {max_diff:.3e} (tol {tol:.0e})"
        )
        if max_diff <= tol:
            dist_utils.rank0_print("verify PASS: 流水线与单进程训练数值一致")
        else:
            dist_utils.rank0_print("verify FAIL: 数值差异超出容差，请检查 loss 缩放语义")

    dist_utils.cleanup()


if __name__ == "__main__":
    main()
