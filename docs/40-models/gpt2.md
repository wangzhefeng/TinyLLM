> 所属：[TinyLLM Docs](../README.md) › [Models](index.md)

# GPT-2

> 所属：[TinyLLM Docs](../README.md) › [Models](index.md)

## 实现

- 模型：`models/gpt2/gpt2_124M.py`（`Model`）
- 配置：`config/gpt2/model_cfgs.py`（124M/355M/774M/1558M）
- 层：`layers/transformer_block.py`、`layers/attention.py`、`layers/feed_forward.py`
- 生成：`layers/inference.py`
- 权重加载：[inference/loaders/gpt2/](../../inference/loaders/gpt2/)

## 结构

- learned token + position embedding
- pre-LN Transformer block（LayerNorm → MHA → residual → LayerNorm → GELU FFN → residual）
- final LayerNorm + 输出头（无 bias）
- 支持 KV cache 的教材实现（`layers/kv_cache.py`）

## 相关

- [预训练 training](../10-pretraining/training.md)
- [示例](../10-pretraining/examples.md)
- [tokenizers](tokenizers.md)
- [LLMs-from-scratch ch04](../60-reference/llms-from-scratch/ch04.md)
