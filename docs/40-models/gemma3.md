> 所属：[TinyLLM Docs](../README.md) › [Models](index.md)

# Gemma3

## 实现

- 模型：`models/gemma3/Gemma3_270M.py`
- 层：`layers/transformer_block.py` 的 `TransformerBlockGemma3`
- 注意力：`layers/attention.py` 的 `GroupedQueryAttention_Gemma3`
- 配置：`config/gemma3/model_cfgs.py`（`GEMMA3_CONFIG_270M`，2026-10-24 从模型 main 内嵌迁出）
- 权重下载/映射：`models/io/gemma3/load_weights.py`

## 结构要点

- 局部/全局 attention 交替（sliding window mask）
- 局部/全局两套 RoPE
- GeGLU 风格 FFN、RMSNorm（4-norm block）
- weight tying：`out_head` 与 `tok_embed` 共享（官方 270m 无独立 lm_head）

## 推断闭环（2026-10-24）

- `inference/registry.py` 的 `gemma3-270m` 条目 `available=True`
  （eos 为 base 版 `<eos>`=1；-it 版应改用 `<end_of_turn>`）
- `inference/loading.py::_load_gemma3`：下载权重 + `ensure_gemma_tokenizer` +
  模型组装，extras 返回 `tokenizer_file_path`
- 权重映射测试：`tests/unit/test_gemma3_loading.py`（合成权重，无网络）

## 相关

- [推断 loaders](../30-inference/loaders.md)
- [LLMs-from-scratch ch05](../60-reference/llms-from-scratch/ch05.md)
- [tokenizers](tokenizers.md)
