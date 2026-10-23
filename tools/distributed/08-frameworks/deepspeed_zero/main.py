# -*- coding: utf-8 -*-

# ***************************************************
# * File        : main.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2026-10-24
# * Version     : 1.0.102402
# * Description : 08 单元：DeepSpeed ZeRO 最小对照示例。与 03-zero 手写实现
# *               对照：优化器状态/梯度分片由 deepspeed.initialize 按
# *               ds_config.json 的 zero_optimization.stage 接管，训练循环
# *               只剩 engine.backward(loss) / engine.step() 两行。
# *               本脚本未安装 deepspeed 时打印安装说明并优雅退出（退出码 0）。
# * Link        : https://www.deepspeed.ai/getting-started/
# * Requirement : torch >= 2.5；deepspeed（pip install deepspeed，
# *               官方支持 Linux；macOS/Windows 建议用 Linux 容器或 WSL2，
# *               ZeRO 多进程训练通常需要 CUDA + NCCL）
# ***************************************************

__all__ = []

# python libraries
import sys
from pathlib import Path

# repo root: tools/distributed/08-frameworks/deepspeed_zero/main.py -> parents[4]
ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

try:
    import deepspeed
except ImportError:
    print("=" * 70)
    print("未检测到 deepspeed，本示例无法运行。")
    print()
    print("安装：pip install deepspeed")
    print("注意：DeepSpeed 官方支持 Linux；ZeRO 多进程训练通常需要")
    print("      CUDA + NCCL，macOS 请使用 Linux 容器或云主机。")
    print()
    print("安装后用以下命令启动（ZeRO-2，见同目录 ds_config.json）：")
    print("  deepspeed --num_gpus 2 \\")
    print("      tools/distributed/08-frameworks/deepspeed_zero/main.py")
    print("=" * 70)
    sys.exit(0)

import torch
import torch.nn.functional as F

from tools.distributed._common.dataset import make_toy_dataloaders
from tools.distributed._common.model import ToyMLP


def evaluate(model_engine, test_loader) -> float:
    """各 rank 持有完整测试集，独立评估精度（梯度已同步，结果一致）。"""
    model_engine.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for features, labels in test_loader:
            features = features.to(model_engine.device)
            labels = labels.to(model_engine.device)
            preds = model_engine(features).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.numel()
    return correct / total


# 测试代码 main 函数
def main():
    import argparse
    parser = argparse.ArgumentParser(description="08-frameworks: DeepSpeed ZeRO 最小示例")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    # deepspeed 启动器会向脚本注入 --local_rank（新版本也读 LOCAL_RANK 环境变量）
    parser.add_argument("--local_rank", type=int, default=-1)
    args = parser.parse_args()

    # 对照 03-zero 的 dist_utils.setup()：
    # deepspeed.initialize 内部完成进程组初始化（deepspeed 启动器注入
    # RANK/WORLD_SIZE/LOCAL_RANK）、模型/优化器按 ZeRO stage 分片包装、
    # 梯度分片规约，无需手写。
    train_loader, test_loader = make_toy_dataloaders(batch_size=args.batch_size)
    model = ToyMLP()

    model_engine, optimizer, train_loader, _ = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=model.parameters(),
        # ds_config.json：ZeRO-2（优化器状态 + 梯度分片）。
        # 改 ZeRO-3：把 zero_optimization.stage 改为 3（再切分模型参数）；
        # 开 CPU offload：在 zero_optimization 中加
        #   "offload_optimizer": {"device": "cpu", "pin_memory": true}
        # 半精度：把 "bf16"/"fp16" 的 enabled 改为 true（CPU 不支持）。
        config=str(Path(__file__).resolve().parent / "ds_config.json"),
    )

    step = 0
    for epoch in range(args.epochs):
        model_engine.train()
        for features, labels in train_loader:
            features = features.to(model_engine.device)
            labels = labels.to(model_engine.device)
            logits = model_engine(features)
            loss = F.cross_entropy(logits, labels)
            # 对照 03-zero 的 loss.backward() + 手动梯度分片同步：
            # engine.backward 内部按 ZeRO stage 做梯度 reduce-scatter。
            model_engine.backward(loss)
            # 对照 03-zero 的 optimizer.step()：分片的优化器状态就地更新。
            model_engine.step()
            if model_engine.global_rank == 0 and step % 10 == 0:
                print(f"step {step:>5d} | loss {loss.item():.4f} | epoch {epoch}")
            step += 1
        acc = evaluate(model_engine, test_loader)
        if model_engine.global_rank == 0:
            print(f"epoch {epoch} | test acc {acc:.4f}")

if __name__ == "__main__":
    main()
