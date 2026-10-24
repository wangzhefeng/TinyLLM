> 所属：[TinyLLM Docs](../README.md) › [Models](index.md)

# Gemma 4（E2B / E4B）

Gemma 4 教学复刻（2026-10-24）。**架构已对照官方 config.json 与 transformers
5.5.0.dev0 源码逐项核实**。

## 实现

- 模型：`models/gemma4/gemma4.py`（复用 `TransformerBlockGemma3`，逐层浅拷贝
  cfg 实现 full 层独立 head_dim；含 `layer_pattern` 与
  `precompute_proportional_rope_params`）
- 配置：`config/gemma4/model_cfgs.py`（`GEMMA4_CONFIG_E2B/E4B` 官方尺寸 +
  `GEMMA4_TINY_CONFIG`）
- 测试：`tests/unit/test_gemma4.py`（15 项）

## 核实到的关键结构（相对 Gemma3 的演进）

- **local:global 交错比两档不同**：E2B 为 4:1（35 层），E4B 为 5:1（42 层），
  均末层强制 full_attention
- **proportional partial RoPE**：sliding 层 theta=1e4 全维旋转；full 层
  theta=1e6、`partial_rotary_factor=0.25`，频率指数按完整 head_dim 归一
  （未旋转维 inv_freq=0）
- full 层独立 `global_head_dim=512`（sliding 层 256，经 per-layer config 覆盖）
- q/k/v 三处 RMSNorm，attention scaling=1.0
- tied embeddings + `final_logit_softcapping=30`

## 教学版未实现（参数保留备查，见模块头注释）

`num_kv_shared_layers` 跨层 KV 共享、`use_double_wide_mlp`（仅 KV 共享层
FFN×2，E2B=True/E4B=False）、PLE per-layer embeddings、attention `v_norm`、
MoE block。

## 相关

- [Gemma3](gemma3.md)
- [LLMs-from-scratch ch05](../60-reference/llms-from-scratch/ch05.md)
