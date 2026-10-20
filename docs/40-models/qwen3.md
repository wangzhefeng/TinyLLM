> 所属：[TinyLLM Docs](../README.md) › [Models](index.md)

# Qwen3

> 所属：[TinyLLM Docs](../README.md) › [Models](index.md)

## 实现

- 模型：`models/qwen3/qwen3.py`、`qwen3_original.py`、`qwen3_optimized.py`、`qwen3_batched.py`
- 配置：`config/qwen3/model_cfgs.py`
- 注意力：`layers/attention.py` 的 `GroupedQueryAttention_Qwen3*`
- 加载：`inference/loaders/qwen3/`（0.6B、30B-A3B、KV cache）

## 结构要点

- RMSNorm + RoPE + GQA + QK norm
- SwiGLU FFN；MoE 版本使用 `layers/moe.py`
- 支持 KV cache 与 batched/streaming 生成

## 相关

- [推断 loaders](../30-inference/loaders.md)
- [LLMs-from-scratch ch05](../60-reference/llms-from-scratch/ch05.md)
- [MoE 实现](../../layers/moe.py)
