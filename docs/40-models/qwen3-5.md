> 所属：[TinyLLM Docs](../README.md) › [Models](index.md)

# Qwen3.5（文本栈）

Qwen3.5 教学复刻（2026-10-24）。**架构已对照官方 config.json 与 transformers
`modular_qwen3_5.py` 逐行核实**。按教学范围只实现文本栈：视觉塔、MTP 头、
mrope 多模态分段与 kernel 化 chunk 扫描明确排除（文本输入下 mrope 退化为
普通 partial RoPE；chunk 扫描与串行递推数学等价）。

## 实现

- 模型：`models/qwen3/qwen3_5.py`（`gated_delta_rule` 递推、`RMSNormGated`、
  `GatedDeltaNet_Qwen3_5`、`compute_partial_rope`、
  `GroupedQueryAttention_Qwen3_5`、`TransformerBlockQwen3_5`、`Model`）
- 配置：`config/qwen3/model_cfgs.py` 的 `Qwen3.5-4B` 分支（官方核实值）
- 测试：`tests/unit/test_qwen3_5.py`（14 项）

## 与 Qwen3 的关键差异（4B 核实值）

- **3:1 linear/full 混合层**：3/4 层为 gated delta net 线性注意力
  （qkv/z/b/a 四投影 + 因果 depthwise conv1d(k=4) + L2 归一 delta rule 递推 +
  SiLU 门控 RMSNorm，推断状态 O(1)）
- full 层加 **sigmoid 输出门控**（q_proj 双宽）与 **partial RoPE（25%，θ=1e7）**
- 全体块 norm 改为**零中心 (1+w) RMSNorm**；权重共享 output head
- KVCache 对两类层透明：full 层存 (k,v)，线性层存 (conv_state, recurrent_state)

## 相关

- [Qwen3](qwen3.md)
- [Gated DeltaNet 通用层](architecture.md)（`layers/gated_delta_rule.py`）
