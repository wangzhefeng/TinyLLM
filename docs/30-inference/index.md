# 推理（Inference）

> 所属：[TinyLLM Docs](../README.md)

整合原 `TinyLLMInference` 与项目内推理代码：权重加载、生成、serving、评测、UI。

## 分页

| 页面 | 内容 |
|---|---|
| [loaders.md](loaders.md) | GPT-2 / Qwen3 / Gemma3 / Llama 权重加载 |
| [serving.md](serving.md) | nano_vllm、API 客户端、服务化 |
| [evaluation.md](evaluation.md) | Ollama/OpenAI 评测、reasoning 评测 |
| [reasoning-scaling.md](reasoning-scaling.md) | 推理时扩展、RL、蒸馏概念 |

## 代码结构

```text
inference/
├── serving/     # nano_vllm
├── api/         # Ollama/OpenAI 客户端
├── evaluation/  # 指令回答评测
├── apps/        # Chainlit UI
├── loaders/     # gpt2 / qwen3 / gemma3 / llama
├── examples/    # GPT-2 生成示例
└── scripts/     # UI / 维护脚本
```

## 入口

```bash
bash inference/scripts/run_app_own.sh
python inference/loaders/qwen3/qwen3_06B_kvcache_load.py
```

## 共享原语

`layers/inference.py`、`layers/kv_cache.py` 保留在 `layers/`：它们是 LLMs-from-scratch ch04/ch05 的教材实现，也是预训练采样所需。`nano_vllm` 负责 serving 级优化，不重复实现教材版 KV cache。

## 相关

- [模型](../40-models/index.md)
- [post-training reasoning](../20-post-training/reasoning.md)
- [测试](../50-development/testing.md)
