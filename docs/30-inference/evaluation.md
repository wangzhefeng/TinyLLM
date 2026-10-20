> 所属：[TinyLLM Docs](../README.md) › [模型推断](index.md)

# 评测

> 所属：[TinyLLM Docs](../README.md) › [模型推断](index.md)

## 指令回答评测

- `inference/evaluation/judge.py`：统一 LLM judge 评分入口（`backend="ollama"|"openai"`）
- `inference/evaluation/ollama_evaluate.py` / `openai_evaluate.py`：兼容 shim，保留原签名并委托 judge.py
- `inference/api/`：Ollama/OpenAI 客户端封装

## 推理评测

- `post_training/reasoning/evaluation/benchmark_based_eval/`：MATH-500、选择题
- `post_training/reasoning/evaluation/judgement_based_eval/`：LLM judge、Elo/Bradley-Terry、leaderboard
- 部分脚本需要 MATH-500 数据集缓存与网络权限

## 相关

- [serving](serving.md)
- [reasoning](../20-post-training/reasoning.md)
- [LLMs-from-scratch ch07 评测分析](../60-reference/llms-from-scratch/ch06-07.md)
