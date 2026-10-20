# 推理（Inference）

> 所属：[TinyLLM Docs](../README.md)

整合原 `TinyLLMInference` 与项目内推理代码：权重加载、生成、serving、评测、UI。

## 分页

| 页面 | 内容 |
|---|---|
| [loaders.md](loaders.md) | GPT-2 / Qwen3 / Gemma3 / Llama 权重加载 |
| [serving.md](serving.md) | 推理原理、API 客户端、服务化 |
| [nano-vllm.md](nano-vllm.md) | 迷你 vLLM 引擎：分页 KV + continuous batching |
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
├── examples/    # GPT-2 生成示例、BERT fill-mask 示例
└── scripts/     # UI / 维护脚本
```

## 入口

```bash
bash inference/scripts/run_app_own.sh
python inference/loaders/qwen3/qwen3_06B_kvcache_load.py
python inference/examples/bert_fill_mask.py --text1 '今天天气[MASK]好'
```

## 共享原语

`layers/inference.py`、`layers/kv_cache.py` 保留在 `layers/`：它们是 LLMs-from-scratch ch04/ch05 的教材实现，也是预训练采样所需。`nano_vllm` 负责 serving 级优化，不重复实现教材版 KV cache。

`layers/inference.py` 提供统一的 `sample_next_token(logits, temperature, top_k, top_p)`（greedy / temperature / top-k / top-p，softmax 数值稳定），所有 `generate*` 变体（GPT-2 / Qwen3 / 批量 / 流式 / 早停）均复用它；单元测试见 `tests/unit/test_inference.py`。

## 相关

- [模型](../40-models/index.md)
- [post-training reasoning](../20-post-training/reasoning.md)
- [测试](../50-development/testing.md)
