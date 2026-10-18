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

## 待补（2026-10-18 盘点，对照 LLMs-from-scratch bonus）

- [ ] **MLA**：`models/deepseek/mla.py` 是空类（ch04/05）
- [ ] **独立 SWA** + 显存估算：目前只有 Gemma3 内嵌 local mask（ch04/06）
- [ ] **Gated DeltaNet**（ch04/08）、**DSA**（ch04/09）、**跨层 KV 共享**（ch04/10）：均无实现
- [ ] 模型 zoo：**OLMo3 / Tiny Aya / Qwen3.5 / Gemma4**（ch05/13、15、16、17）
- 验收口径：KV cache 与无 cache greedy 输出逐 token 一致；显存估算与公式一致；能与 HF 权重对齐前向 logits

## 相关

- [预训练](../10-pretraining/index.md)
- [推理 loaders](../30-inference/loaders.md)
