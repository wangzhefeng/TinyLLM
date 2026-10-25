#!/bin/bash

# *********************************************
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Description : 启动 Chainlit UI（合并原 run_app_orig.sh / run_app_own.sh）
#               用法: bash inference/scripts/run_app.sh [orig|own] [gpt2|qwen3]  （默认 own gpt2）
#                 - 第 1 参数（仅 APP_MODEL=gpt2 时生效）：
#                   - orig：官方 GPT-2 124M 权重（下载到 downloaded_models/gpt2）
#                   - own ：自训 checkpoint（用 GPT2_CHECKPOINT 覆盖路径，默认 results/checkpoints/gpt2/model.pth）
#                 - 第 2 参数：模型选择（APP_MODEL 环境变量可预置，命令行参数优先）
#                   - gpt2：GPT-2 124M（现状行为不变）
#                   - qwen3：经 inference.loading.load_model 加载
#                     （默认 key qwen3-0.6b，QWEN3_MODEL_KEY 可覆盖），
#                     UI 带 thinking 开关，流式输出并分离 <think> 段
# * Requirement : pip install chainlit（可选依赖，不在主 requirements 中）；
#               gpt2 orig 模式另需 tensorflow（仅读官方 TF checkpoint 时需要）
# **********************************************

MODE="${1:-own}"
MODEL="${2:-${APP_MODEL:-gpt2}}"
case "$MODE" in
  orig|own) ;;
  *) echo "Usage: $0 [orig|own] [gpt2|qwen3]  (default: own gpt2)" >&2; exit 1 ;;
esac
case "$MODEL" in
  gpt2|qwen3) ;;
  *) echo "Usage: $0 [orig|own] [gpt2|qwen3]  (default: own gpt2)" >&2; exit 1 ;;
esac

# 无论从哪里调用，都切到仓根目录再启动（app.py 内部路径已按仓根锚定，
# chainlit 的运行产物 .chainlit/ 也统一落在仓根，已被 .gitignore 忽略）
cd "$(dirname "$0")/../.." || exit 1
APP_MODEL="$MODEL" GPT2_WEIGHTS="$MODE" chainlit run inference/apps/app.py
