> 所属：[TinyLLM Docs](../README.md) › [Models](index.md)

# Gemma3

> 所属：[TinyLLM Docs](../README.md) › [Models](index.md)

## 实现

- 模型：`models/gemma3/Gemma3_270M.py`
- 层：`layers/transformer_block.py` 的 `TransformerBlockGemma3`
- 注意力：`layers/attention.py` 的 `GroupedQueryAttention_Gemma3`
- 加载：`inference/loaders/gemma3/load_weights.py`

## 结构要点

- 局部/全局 attention 交替（sliding window mask）
- 局部/全局两套 RoPE
- GeGLU 风格 FFN、RMSNorm
- 权重加载使用 safetensors

## 相关

- [推断 loaders](../30-inference/loaders.md)
- [LLMs-from-scratch ch05](../60-reference/llms-from-scratch/ch05.md)
- [tokenizers](tokenizers.md)
