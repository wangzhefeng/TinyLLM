> 所属：[TinyLLM Docs](../README.md) › [Models](index.md)

# Phi-1.5

phi-1.5（microsoft/phi-1_5，1.3B）教学实现，对齐 rasbt/LLMs-from-scratch ch07
的指令微调底座路线（2026-10-24）。

## 实现

- 模型：`models/phi/phi15.py`（`PhiAttention` / `PhiMLP` / `PhiBlock` / `Model`）
- 配置：`config/phi/model_cfgs.py`（`PHI15_CONFIG` 官方尺寸 + `PHI15_TINY_CONFIG`）
- 权重下载/映射：`models/io/phi/load_phi15_weights.py`
- 注册：`models/registry.py` 的 `phi-15`（eos=50256）

## 结构要点（对照 HF config.json）

- MHA（32 头，head_dim=64），带 bias 的 Q/K/V/out 投影
- **partial RoPE**：`partial_rotary_factor=0.5`，位置旋转只作用于每头前
  32 维，后 32 维直通（`PhiAttention.rotary_dim`）
- LayerNorm（eps=1e-5，weight+bias）+ gelu_new（tanh）FFN（8192）
- GPT-2 式串行残差；无绝对位置编码、无 KV cache（教学版）
- vocab 51200（含 [PAD]=50257 等扩展 tokens），ctx 2048，不 tying

## 权重映射

- HF checkpoint 的合并 `self_attn.qkv_proj` (3d, d) 按行拆三份注入
  `W_query`/`W_key`/`W_value`（bias 同步拆分），无需转置
- LayerNorm 的 weight/bias 映射到自定义 LayerNorm 的 `scale`/`shift`
- 仅支持 HF PhiForCausalLM 命名；缺 `lm_head.weight` 时回退 weight tying
- 测试：`tests/unit/test_phi15_loading.py`（合成权重，无网络）

## 相关

- [权重 IO](../30-inference/loaders.md)
- [LLMs-from-scratch ch06-07](../60-reference/llms-from-scratch/ch06-07.md)
