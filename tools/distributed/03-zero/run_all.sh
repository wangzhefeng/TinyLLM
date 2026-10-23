#!/usr/bin/env bash
# 03-zero：依次用 torchrun 2 进程跑 ZeRO stage 0/1/2/3，最后跑等价性验证。
# 从仓库根目录运行（脚本会自动 cd 到仓库根，与调用时的 cwd 无关）。
# CPU + gloo 即可；stage 0 是 DDP 等价基线，1/2/3 为 ZeRO 三阶段手写实现。
set -euo pipefail

cd "$(dirname "$0")/../../.."
PY=.venv/bin/python
NPROC=2
BASE_PORT=29600

for STAGE in 0 1 2 3; do
    echo "==================== ZeRO stage ${STAGE} ===================="
    $PY -m torch.distributed.run \
        --nproc_per_node=$NPROC \
        --master_port=$((BASE_PORT + STAGE)) \
        tools/distributed/03-zero/train.py --stage $STAGE --epochs 1
done

echo "==================== 等价性验证（单进程基线 vs 各 stage） ===================="
$PY tools/distributed/03-zero/verify_equivalence.py
