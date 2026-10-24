> 所属：[TinyLLM Docs](../README.md) › [Models](index.md)

# Tiny Aya（260M，Cohere Command 家族）

Tiny Aya 教学复刻（2026-10-24）。⚠️ **架构未经官方 config 核实**——
`cohere-for-ai/tinyaya` 为门控模型（HTTP 401），按 Command 家族公开特性集
实现；260M 超参为量级推算的假设，待官方 config 可访问后校正。

## 实现

- 模型：`models/aya/aya.py`（`ComplexAffinePositionEncoding` +
  `GroupedQueryAttentionAya` + `TransformerBlockAya` + `Model`）
- 配置：`config/aya/model_cfgs.py`（`AYA_CONFIG_260M`（假设）+
  `AYA_TINY_CONFIG`）
- 测试：`tests/unit/test_aya.py`（15 项）

## CAT 复合仿射位置编码

head_dim 相邻两维配对为复数 `z = x[2k] + i·x[2k+1]`，乘 `exp(i·θ_t)`
（θ 频率与 RoPE 同型），再做每维可学习仿射 `y·(1+scale) + shift`；
scale/shift **零初始化** → 初始即纯旋转：位置 0 精确恒等、任意位置保范数、
内积只依赖相对位置（三性质均有测试）。

## uvqk 注意力偏置

四组每头每维偏置。**实现要点**：纯加性 logit 偏置的 query 侧项对每行是
常数、会被 softmax 整行消去（数学必然），故采用**向量注入**——q/k 偏置在
KV expand 后、CAT 旋转前注入未旋转空间，v_bias 注入 value，u_bias 注入每头
注意力输出；零初始化时精确等价于标准缩放点积 GQA（有手工复算对照测试），
四组逐一扰动均改变输出。

## 其他

GQA（12:4 假设）、SwiGLU、RMSNorm、weight tying（out_head 与 tok_embed 共享）。

## 相关

- [LLMs-from-scratch ch05](../60-reference/llms-from-scratch/ch05.md)
