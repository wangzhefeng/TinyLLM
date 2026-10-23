#!/usr/bin/env bash
# 06-pp：torch.distributed.pipelining 流水线并行（从仓库根目录运行，路径与 cwd 无关）
# 2 进程 = 2 stage（toy Transformer 每层一个 stage），CPU + gloo 即可运行。
# 注意：torch 2.5.1 的 Schedule1F1B 在 CPU + gloo 下有框架级 bug（见 main.py 注释），
# CPU 环境请用默认的 gpipe；1f1b 需 CUDA + NCCL。
# 可选参数透传给 main.py，例如：
#   bash tools/distributed/06-pp/run.sh --schedule gpipe --n-microbatches 8
#   bash tools/distributed/06-pp/run.sh --verify
NUM_TRAINERS=2

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$REPO_ROOT"

# 优先使用仓库 .venv 里的 torchrun，否则回退到 PATH
TORCHRUN=torchrun
if [ -x "$REPO_ROOT/.venv/bin/torchrun" ]; then
    TORCHRUN="$REPO_ROOT/.venv/bin/torchrun"
fi

"$TORCHRUN" \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=$NUM_TRAINERS \
    tools/distributed/06-pp/main.py "$@"
