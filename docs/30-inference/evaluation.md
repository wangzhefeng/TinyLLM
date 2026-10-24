> 所属：[TinyLLM Docs](../README.md) › [模型推断](index.md)

# 评测

## 指令回答评测

- `inference/evaluation/judge.py`：统一 LLM judge 入口，backend 为函数参数 `ollama` 或 `openai`。
- `ollama_evaluate.py` / `openai_evaluate.py`：保留旧签名的兼容 shim。
- `inference/api/`：外部服务客户端；OpenAI 使用 `OPENAI_API_KEY` 环境变量。

## 推理评测

| 目录 | 内容 |
|---|---|
| `inference/evaluation/benchmarks/` | MATH-500 答案解析、MMLU 选择题与 logprob 评测 |
| `inference/evaluation/judges/` | Ollama/Qwen3 judge、Elo、Bradley-Terry、leaderboard |

旧 `post_training/reasoning/evaluation/` 路径保留兼容 shim。数据下载和模型加载在调用时发生，核心纯函数可离线单测；MATH/MMLU 与外部 judge 需要对应缓存、网络或服务。

## 相关

- [serving](serving.md)
- [reasoning](../20-post-training/reasoning.md)
- [数据集](../50-development/datasets.md)
