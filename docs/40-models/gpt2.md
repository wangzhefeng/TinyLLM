> 所属：[TinyLLM Docs](../README.md) › [Models](index.md)

# GPT-2

> 所属：[TinyLLM Docs](../README.md) › [Models](index.md)

## 实现

- 模型：`models/gpt2/gpt2_124M.py`（`Model`）
- 配置：`config/gpt2/model_cfgs.py`（124M/355M/774M/1558M）
- 层：`layers/transformer_block.py`、`layers/attention.py`、`layers/feed_forward.py`
- 生成：`inference/generate_core.py`
- 权重加载：[`models/io/gpt2/`](../../models/io/gpt2/)

## 结构

- learned token + position embedding
- pre-LN Transformer block（LayerNorm → MHA → residual → LayerNorm → GELU FFN → residual）
- final LayerNorm + 输出头（无 bias）
- 支持 KV cache 的教材实现（`layers/kv_cache.py`）

## 与 HF 对齐

`tests/unit/test_gpt2_hf_alignment.py` 断言手写实现与 `transformers` 官方 GPT-2 的**前向输出**一致：随机权重（tiny 配置，CI 常跑）logits 逐位对齐 + 逐步贪心 argmax 一致；本地缓存存在时对真实 124M 权重对齐 logits（atol 1e-4）与贪心续写。与 [test_gpt2_loading](../50-development/testing.md)（键名映射）互补。

对齐依赖的三个关键细节：

1. **因果掩码**：每个位置只能看到自己及之前的位置（`layers/attention.py` 的 `causal_mask`）；
2. **权重转置**：官方/HF 的 Conv1D 权重是 `[in, out]`，`nn.Linear` 是 `[out, in]`，加载需转置；`c_attn` 把 Q/K/V 拼在一起，需按最后一维切三份（`models/io/gpt2/openai_gpt2_weights_load_hf.py`）；
3. **GELU 版本**：GPT-2 用 tanh 近似版，须 `nn.GELU(approximate="tanh")`（`layers/feed_forward.py`），PyTorch 默认精确版会引入数值偏差。

## 相关

- [预训练 training](../10-pretraining/training.md)
- [示例](../10-pretraining/examples.md)
- [tokenizers](tokenizers.md)
- [LLMs-from-scratch ch04](../60-reference/llms-from-scratch/ch04.md)
