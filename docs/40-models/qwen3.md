> 所属：[TinyLLM Docs](../README.md) › [Models](index.md)

# Qwen3

> 所属：[TinyLLM Docs](../README.md) › [Models](index.md)

## 实现

- 模型：`models/qwen3/qwen3.py`（基础，含 KV cache）、`qwen3_optimized.py`（SDPA + 预分配 cache）、`qwen3_batched.py`（batch + padding mask + 逐序列早停）
- 配置：`config/qwen3/model_cfgs.py`
- 注意力：`layers/attention.py` 的 `GroupedQueryAttention_Qwen3*`
- 加载：`inference/loaders/qwen3/`（0.6B、30B-A3B、KV cache）

> 2026-10-20 合并：`qwen3_original.py`（与 `qwen3.py` 逐字重复，仅属性名不同）已删除。
> 加载书本 checkpoint（rasbt/qwen3-from-scratch，tok_emb/att 命名）时用
> `inference/loaders/qwen3/load_weights.py:remap_book_state_dict` 重映射后 `load_state_dict`。

## 结构要点

- RMSNorm + RoPE + GQA + QK norm
- SwiGLU FFN；MoE 版本使用 `layers/moe.py`
- 支持 KV cache 与 batched/streaming 生成；计算精度跟随 `cfg.dtype`（不再硬编码 bf16）
- RoPE 越界防护：cache 位置超过 `context_length` 时抛出明确的 IndexError

## 相关

- [推断 loaders](../30-inference/loaders.md)
- [LLMs-from-scratch ch05](../60-reference/llms-from-scratch/ch05.md)
- [MoE 实现](../../layers/moe.py)
