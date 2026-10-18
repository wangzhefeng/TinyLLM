> 所属：[TinyLLM Docs](../README.md) › [Inference](index.md)

# 评测

> 所属：[TinyLLM Docs](../README.md) › [Inference](index.md)

## 指令回答评测

- `inference/evaluation/ollama_evaluate.py`：本地 Ollama 模型打分
- `inference/evaluation/openai_evaluate.py`：OpenAI API 打分
- `inference/api/`：Ollama/OpenAI 客户端封装

## Reasoning 评测

- `post_training/reasoning/evaluation/benchmark_based_eval/`：MATH-500、选择题
- `post_training/reasoning/evaluation/judgement_based_eval/`：LLM judge、Elo/Bradley-Terry、leaderboard
- 部分脚本需要 MATH-500 数据集缓存与网络权限

## 相关

- [serving](serving.md)
- [reasoning](../20-post-training/reasoning.md)
- [LLMs-from-scratch ch07 评测分析](../60-reference/llms-from-scratch/ch06-07.md)
