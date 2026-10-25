# 模型

> 所属：[TinyLLM Docs](../README.md)

共享模型定义与权重加载说明。模型代码在 `models/`，注意力/FFN/norm 等共享层在 `layers/`。

## 页面

| 模型 | 文档 | 代码 |
|---|---|---|
| GPT-2 | [gpt2.md](gpt2.md) | `models/gpt2/` |
| Llama 2/3 | [llama.md](llama.md) | `models/llama/` |
| Qwen3 | [qwen3.md](qwen3.md) | `models/qwen3/` |
| Qwen3.5 | [qwen3-5.md](qwen3-5.md) | `models/qwen3/qwen3_5.py` |
| Olmo 3 | [olmo3.md](olmo3.md) | `models/olmo3/` |
| Gemma3 | [gemma3.md](gemma3.md) | `models/gemma3/` |
| Gemma 4 | [gemma4.md](gemma4.md) | `models/gemma4/` |
| Tiny Aya | [aya.md](aya.md) | `models/aya/` |
| BERT | [bert.md](bert.md)、[实验记录](bert-experiments.md) | `models/bert/` |
| DeepSeek | [deepseek.md](deepseek.md) | `models/deepseek/` |
| Phi-1.5 | [phi.md](phi.md) | `models/phi/` |
| Tokenizer | [tokenizers.md](tokenizers.md) | `layers/tokenizers/` |

## 架构与背景

- [LLM 架构笔记](architecture.md)：Transformer/注意力/位置编码/变体
- [Karpathy 来源分析](karpathy-provenance.md)：nanoGPT/nanochat/minGPT 对照（副本已删除）

## 待补（2026-10-18 盘点，2026-10-24 更新，对照 LLMs-from-scratch bonus）

- [x] **MLA**：`models/deepseek/mla.py`（2026-10-20 实现完整版）
- [x] **独立 SWA**：`layers/attention.py::SlidingWindowAttention` + `trim_kv_cache_window`
- [x] **跨层 KV 共享**：`layers/kv_cache.py::SharedKVCache` + Qwen3 `kv_sharing_span` 接线（2026-10-24）
- [x] **DSA 稀疏注意力**：`layers/sparse_attention.py::NativeSparseAttention`（NSA 三分支 + 门控，2026-10-24）
- [x] **Gated DeltaNet**：`layers/gated_delta_rule.py::GatedDeltaNet`（recurrent + chunked 精确并行等价，2026-10-24）
- [x] 模型 zoo（2026-10-24）：**Qwen3.5**（官方 config 核实）与 **Gemma4**（官方
  config 核实）已复刻；**Olmo3** 经 transformers 源码核实（checkpoint 门控无法
  直接访问）；**TinyAya** 架构未核实（门控 401，按 Command 家族特性实现并标注）
- 验收口径：KV cache 与无 cache greedy 输出逐 token 一致；显存估算与公式一致；能与 HF 权重对齐前向 logits

## 相关

- [预训练](../10-pretraining/index.md)
- [权重 IO](../30-inference/loaders.md)
