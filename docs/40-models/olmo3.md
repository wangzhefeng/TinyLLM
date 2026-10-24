> 所属：[TinyLLM Docs](../README.md) › [Models](index.md)

# Olmo 3

Olmo 3（AllenAI）教学复刻（2026-10-24）。⚠️ checkpoint 级 config 未能直接核实
（HF 权重库 401 gated），但**结构已对照 transformers 官方源码核实**
（`configuration_olmo3.py` / `modular_olmo3.py` / `modeling_olmo2.py`）；
"7B" 尺寸取 `Olmo3Config` 类默认值（context_length 取模型卡公开的 65,536）。

## 实现

- 模型：`models/olmo3/olmo3.py`（`NonParametricRMSNorm`、
  `GroupedQueryAttention_Olmo3`、`TransformerBlockOlmo3`、`Model`，含 KV cache
  增量与逐层 full/滑窗 mask）
- 配置：`config/olmo3/model_cfgs.py`
- 测试：`tests/unit/test_olmo3.py`（12 项）

## 与 Llama/Qwen3 的关键差异

- **post-norm 块**：attn/FFN 输出先 norm 再残差相加（Llama/Qwen3 为 pre-norm）
- **非参数化 QK-norm**：作用于整个投影输出向量（Qwen3 为逐 head 带权重）
- **3:1 滑窗/full 混合层**：`sliding_window=4096`，`(i+1)%4==0` 为 full 层
- RoPE θ=500000（同 Llama3）；MHA（7B 级默认 32 头无 GQA）

## 相关

- [Qwen3](qwen3.md)
- [LLMs-from-scratch ch05](../60-reference/llms-from-scratch/ch05.md)
