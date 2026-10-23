# -*- coding: utf-8 -*-

# ***************************************************
# * File        : train.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-08-20
# * Version     : 1.0.082013
# * Description : 04 单元：FSDP2（fully_shard）toy Transformer 训练示例。
# *               逐层 fully_shard + 可选显式预取 + 可选混合精度
# *               + 两种 checkpoint API（dtensor_api / dcp_api）演示。
# *               checkpoint 固定写入仓库根的
# *               results/checkpoints/distributed/fsdp/（与 cwd 无关）。
# *               CUDA-only：fully_shard 依赖 CUDA stream，无 GPU 时入口
# *               明确报错退出。
# * Link        : https://github.com/pytorch/examples/tree/main/distributed/FSDP2
# * Requirement : torch >= 2.5；CUDA GPU + nccl
# ***************************************************

import argparse
import os
import sys
from pathlib import Path

# repo root: tools/distributed/04-fsdp/train.py -> parents[3]
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import torch
from checkpoint import Checkpointer
from model import ModelArgs, Transformer
from inspect_utils import inspect_mixed_precision, inspect_model

# FSDP2 导入：torch >= 2.6 在 torch.distributed.fsdp 下，2.5 在 _composable 下
try:
    from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
except ImportError:
    from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy

from tools.distributed._common.dist import get_checkpoint_dir, rank0_print


def set_modules_to_forward_prefetch(model, num_to_forward_prefetch):
    for i, layer in enumerate(model.layers):
        if i >= len(model.layers) - num_to_forward_prefetch:
            break
        layers_to_prefetch = [
            model.layers[i + j] for j in range(1, num_to_forward_prefetch + 1)
        ]
        layer.set_modules_to_forward_prefetch(layers_to_prefetch)


def set_modules_to_backward_prefetch(model, num_to_backward_prefetch):
    for i, layer in enumerate(model.layers):
        if i < num_to_backward_prefetch:
            continue
        layers_to_prefetch = [
            model.layers[i - j] for j in range(1, num_to_backward_prefetch + 1)
        ]
        layer.set_modules_to_backward_prefetch(layers_to_prefetch)


def main(args):
    # FSDP2（fully_shard）在 torch 2.5/2.6 依赖 CUDA stream 做通信重叠，
    # CPU + gloo 下会直接报 "FSDP requires CUDA for streams"，因此本单元
    # 为 CUDA-only：入口处明确报错，不做隐晦的 CPU 回退。
    if not torch.cuda.is_available():
        raise SystemExit(
            "04-fsdp 需要 CUDA GPU：FSDP2(fully_shard) 依赖 CUDA stream，"
            "不支持 CPU/MPS。请在 GPU 机器上运行（见 04-fsdp/run*.sh），"
            "CPU 上的分布式冒烟请改用 02-ddp。"
        )
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    torch.distributed.init_process_group(backend="nccl", device_id=device)
    torch.manual_seed(0)
    vocab_size = 1024
    batch_size = 32
    seq_len = 64
    model_args = ModelArgs(
        n_layers=10,
        n_heads=4,
        vocab_size=vocab_size,
        max_seq_len=seq_len,
        dropout_p=0,
    )
    with torch.device("meta"):
        model = Transformer(model_args)
    fsdp_kwargs = {}
    if args.mixed_precision:
        fsdp_kwargs["mp_policy"] = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
        )
    for layer in model.layers:
        fully_shard(layer, **fsdp_kwargs)
    fully_shard(model, **fsdp_kwargs)

    inspect_model(model)

    if args.explicit_prefetch:
        set_modules_to_forward_prefetch(model, num_to_forward_prefetch=2)
        set_modules_to_backward_prefetch(model, num_to_backward_prefetch=2)

    # checkpoint 目录固定为仓库根下 results/checkpoints/distributed/fsdp/
    checkpointer = Checkpointer(str(get_checkpoint_dir("fsdp")), dcp_api=args.dcp_api)
    if checkpointer.last_training_time is None:
        model.to_empty(device=device)
        model.reset_parameters()
    else:
        checkpointer.load_model(model)

    if args.mixed_precision:
        inspect_mixed_precision(model)

    optim = torch.optim.Adam(model.parameters(), lr=1e-2)
    if checkpointer.last_training_time is not None:
        checkpointer.load_optim(model, optim)

    for step in range(args.steps):
        if args.explicit_prefetch:
            model.unshard()
        x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        loss = model(x).sum()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optim.step()
        optim.zero_grad()
        if step % args.log_every == 0:
            rank0_print(f"step {step:>4d} | loss {loss.item():.4f}")

    checkpointer.save(model, optim)
    rank0_print(f"checkpoint saved under {checkpointer.folder}")
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch FSDP2 example")
    parser.add_argument("--explicit-prefetch", action="store_true", default=False)
    parser.add_argument("--mixed-precision", action="store_true", default=False)
    parser.add_argument("--dcp-api", action="store_true", default=False)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--log-every", type=int, default=1)
    args = parser.parse_args()
    main(args)
