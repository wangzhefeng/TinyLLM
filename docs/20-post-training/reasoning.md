> 所属：[TinyLLM Docs](../README.md) › [Post-training](index.md)

# Reasoning

对应 [reasoning-from-scratch](https://github.com/rasbt/reasoning-from-scratch) 的实现目录。

> **归属说明**：reasoning **训练**（RLVR/GRPO、蒸馏、reasoning SFT）属于后训练；  
> **推理时扩展**（CoT、self-consistency、best-of-n、self-refinement）和 **reasoning 评测**更接近推理域，  
> 目前为了对照书本内容暂放在本目录，后续可拆分到 `inference/`。

## 目录

| 目录 | 内容 | 状态 |
|---|---|---|
| `evaluation/benchmark_based_eval/` | MATH-500 答案解析、选择题评测 | 部分可用 |
| `evaluation/judgement_based_eval/` | LLM judge、Elo/Bradley-Terry、leaderboard | 多为占位 |
| `training/` | GRPO、蒸馏 | 空壳，待补 |
| `inference/` | CoT、self-consistency、best-of-n、self-refinement、Qwen3 demo | 部分占位 |

## 说明

- 评测脚本会下载/读取 MATH-500，需要 Hugging Face 数据集缓存与网络权限。
- `inference/llm_basic/qwen3_06B.py` 在导入时就会加载模型，适合作为脚本运行，不宜直接 import 到单元测试。
- 推理时扩展（CoT、self-consistency 等）目前留在本目录；训练/推理/评测已分别放在 `training/`、`inference/`、`evaluation/`。
