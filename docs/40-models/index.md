# 模型

> 所属：[TinyLLM Docs](../README.md)

共享模型定义与权重加载说明。模型代码在 `models/`，注意力/FFN/norm 等共享层在 `layers/`。

## 页面

| 模型 | 文档 | 代码 |
|---|---|---|
| GPT-2 | [gpt2.md](gpt2.md) | `models/gpt2/` |
| Llama 2/3 | [llama.md](llama.md) | `models/llama/` |
| Qwen3 | [qwen3.md](qwen3.md) | `models/qwen3/` |
| Gemma3 | [gemma3.md](gemma3.md) | `models/gemma3/` |
| BERT | [bert.md](bert.md)、[实验记录](bert-experiments.md) | `models/bert/` |
| DeepSeek | [deepseek.md](deepseek.md) | `models/deepseek/` |
| Tokenizer | [tokenizers.md](tokenizers.md) | `layers/tokenizers/` |

## 架构与背景

- [LLM 架构笔记](architecture.md)：Transformer/注意力/位置编码/变体
- [Karpathy 来源分析](karpathy-provenance.md)：nanoGPT/nanochat/minGPT 对照（副本已删除）

## 相关

- [预训练](../10-pretraining/index.md)
- [推理 loaders](../30-inference/loaders.md)
