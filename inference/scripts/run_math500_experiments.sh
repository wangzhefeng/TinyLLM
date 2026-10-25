#!/bin/bash

# *********************************************
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Description : MATH-500 端到端评测批量脚本（对应书 run_all_experiments_math500.sh）：
# *               依次以 greedy/cot/self-consistency/best-of-n/self-refinement
# *               跑 --subset 50，各策略报告落盘 results/math500/，末尾汇总 accuracy
# *               用法: bash inference/scripts/run_math500_experiments.sh
# *               可用环境变量覆盖：MODEL / SUBSET / N / TEMPERATURE /
# *               BATCH_SIZE / OUTDIR / PYTHON
# * Requirement : torch + 本地 MATH-500 缓存（dataset/evaluation/math500/）
# **********************************************

set -euo pipefail

# 无论从哪里调用，都切到仓根目录（harness 的数据缓存路径按仓根锚定）
cd "$(dirname "$0")/../.." || exit 1

PYTHON="${PYTHON:-.venv/bin/python}"
MODEL="${MODEL:-qwen3-0.6b}"
SUBSET="${SUBSET:-50}"
N="${N:-5}"
TEMPERATURE="${TEMPERATURE:-0.8}"
BATCH_SIZE="${BATCH_SIZE:-4}"
OUTDIR="${OUTDIR:-results/math500}"

mkdir -p "$OUTDIR"
MAIN_LOG="$OUTDIR/all_runs.txt"

echo "=== MATH-500 batch start: $(date -Is)" | tee "$MAIN_LOG"
echo "=== MODEL: $MODEL  SUBSET: $SUBSET  N: $N  OUTDIR: $OUTDIR" | tee -a "$MAIN_LOG"
echo | tee -a "$MAIN_LOG"

for STRATEGY in greedy cot self-consistency best-of-n self-refinement; do
    OUT_JSON="$OUTDIR/${MODEL}_${STRATEGY}_subset${SUBSET}.json"
    echo "=== strategy: $STRATEGY -> $OUT_JSON" | tee -a "$MAIN_LOG"
    echo "Command: $PYTHON -m inference.harness.math500 --model $MODEL --strategy $STRATEGY --n $N --subset $SUBSET" | tee -a "$MAIN_LOG"

    PYTHONUNBUFFERED=1 "$PYTHON" -m inference.harness.math500 \
        --model "$MODEL" \
        --strategy "$STRATEGY" \
        --n "$N" \
        --subset "$SUBSET" \
        --batch-size "$BATCH_SIZE" \
        --temperature "$TEMPERATURE" \
        --out "$OUT_JSON" 2>&1 | tee -a "$MAIN_LOG"

    echo | tee -a "$MAIN_LOG"
done

echo "=== accuracy summary" | tee -a "$MAIN_LOG"
"$PYTHON" - "$OUTDIR" <<'EOF' | tee -a "$MAIN_LOG"
import json
import sys
from pathlib import Path

outdir = Path(sys.argv[1])
for path in sorted(outdir.glob("*_subset*.json")):
    report = json.loads(path.read_text(encoding="utf-8"))
    acc = report["accuracy"]
    n = report["num_problems"]
    print(f"{path.stem}: accuracy={acc * 100:.1f}% ({round(acc * n)}/{n} problems)")
EOF

echo "=== MATH-500 batch done: $(date -Is)" | tee -a "$MAIN_LOG"
