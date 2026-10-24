#!/bin/bash

# *********************************************
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Description : 启动 Chainlit UI（合并原 run_app_orig.sh / run_app_own.sh）
#               用法: bash inference/scripts/run_app.sh [orig|own]  （默认 own）
#                 - orig：官方 GPT-2 124M 权重（下载到 downloaded_models/gpt2）
#                 - own ：自训 checkpoint（用 GPT2_CHECKPOINT 覆盖路径，默认仓根下 model.pth）
# * Requirement : pip install chainlit（可选依赖，不在主 requirements 中）；
#               orig 模式另需 tensorflow（仅读官方 TF checkpoint 时需要）
# **********************************************

MODE="${1:-own}"
case "$MODE" in
  orig|own) ;;
  *) echo "Usage: $0 [orig|own]  (default: own)" >&2; exit 1 ;;
esac

# 无论从哪里调用，都切到仓根目录再启动（app.py 内部路径已按仓根锚定，
# chainlit 的运行产物 .chainlit/ 也统一落在仓根，已被 .gitignore 忽略）
cd "$(dirname "$0")/../.." || exit 1
GPT2_WEIGHTS="$MODE" chainlit run inference/apps/app.py
