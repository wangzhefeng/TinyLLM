# 缺失主题清单（对照两本上游书）

> 所属：[TinyLLM Docs](../../README.md) › [参考与分析](../index.md) › [layers/models 审查](index.md)

按预训练 / 推断 / 后训练三个视角，列出 `layers/` 与 `models/` 相对
[LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch) 与
[reasoning-from-scratch](https://github.com/rasbt/reasoning-from-scratch) 缺失的主题。
✅ = 已具备；❌ = 缺失；🟡 = 有零件但未闭环。

## 1. 预训练视角

### 1.1 注意力与位置编码（layers/）

| 主题 | 状态 | 说明 |
|---|---|---|
| MHA / GQA / SDPA / FlexAttention | ✅ | `layers/attention.py` 8 种实现 |
| MLA（DeepSeek 低秩 latent KV） | ❌ | `models/deepseek/mla.py` 仅 NotImplementedError 骨架 |
| 独立 SWA 层 | 🟡 | 滑窗逻辑锁死在 Gemma3 block 内，layers/ 无可复用 SWA |
| 跨层 KV 共享 | ❌ | KV cache 不支持跨层复用 |
| Gated DeltaNet / DSA | ❌ | 线性注意力与稀疏注意力均缺 |
| RoPE 长上下文扩展 | ❌ | 仅 Llama3 式分段缩放；无 NTK-aware / YaRN / 动态外推 |
| ALiBi | ❌ | 与可学习 PE、RoPE 构成三足对照，成本极低 |
| pre-norm / post-norm 可切换 | ❌ | block 全部硬编码 pre-norm，无法做经典消融 |

### 1.2 FFN / MoE / 归一化

| 主题 | 状态 | 说明 |
|---|---|---|
| SwiGLU | ✅ | `FeedForwardSiLU` 即 SwiGLU FFN；`FeedForwardSwiGLU`/`SwiGLU` 空壳已在 Phase 0 补齐实现 |
| MoE 负载均衡 | ❌ | 无 aux loss / router z-loss；超容量 token 静默丢弃且无统计 |
| DeepSeekMoE（shared + routed expert） | ❌ | `layers/moe.py` 是 Qwen3 式 MoE |
| RMSNorm fp32 上转一致性 | 🟡 | 基础版无上转，Qwen3/Gemma3 版有，语义不统一 |

### 1.3 模型 zoo（models/）

| 模型 | 状态 | 说明 |
|---|---|---|
| GPT-2 / Llama2 / Llama3 / Qwen3 / Gemma3 / BERT | ✅ | 可构造（部分有 P0 bug，见 bugs.md） |
| Llama3.1 / 3.2 | ❌ | `llama31_8B.py`、`llama32_1B.py` 是空壳（只有 `main(): pass`），config 与 GQA 已支持 rope_freq，可直接复用 Llama3 |
| DeepSeek-V2/V3（MLA + MoE + MTP） | ❌ | 整体骨架 |
| Mistral / Mixtral | ❌ | SWA 最简形态 + top-2 MoE 教科书案例 |
| OLMo3 / Tiny Aya / Qwen3.5 / Gemma4 / GPT-OSS | ❌ | 上游 ch05 bonus 模型 |
| T5（encoder-decoder） | ❌ | 已有 encoder-only（BERT）与 decoder-only，缺中间形态 |

## 2. 推断视角

| 主题 | 状态 | 说明 |
|---|---|---|
| greedy / temperature / top-k / top-p | ✅ | `inference/generate_core.py:sample_next_token` |
| beam search | ❌ | 无 |
| repetition / frequency / presence penalty、min-p | ❌ | 无 |
| KV cache（朴素 + 预分配） | ✅ | 两种实现，但接口互不兼容 |
| batched 生成 + 逐序列 early stop | ✅ | `generate_text_basic_batched_*_stop` |
| streaming | ✅ | yield 式 |
| 推断时扩展（reasoning ch04/05） | ❌ | CoT 脚手架、self-consistency 投票、self-refinement、Best-of-N 均为占位 |
| KV cache × SWA 窗口截断 / ×MLA latent | ❌ | cache 无窗口与 latent 概念 |
| 量化（int8/FP8/GGUF）、投机解码 | ❌ | 无 |
| paged KV cache | 🟡 | `inference/serving` 的 nano-vllm 与 layers/ 的 cache 是两套无关联实现 |

## 3. 后训练视角

| 主题 | 状态 | 说明 |
|---|---|---|
| LoRA / DoRA | ✅ | `post_training/peft/layers/lora_dora.py`（但 layers/ 无通用 LoRA 层，models/ 不能即插即用） |
| 分类头（ch06） | 🟡 | `post_training/sft/layers/clf_head.py` 存在，但 models/ 各模型无 `num_classes` 输出模式 |
| DPO | ✅ | `post_training/preference/exp/exp_finetune_gpt_dpo.py` |
| GRPO（unsloth/vllm 版） | 🟡 | `post_training/preference/model_rlhf/grpo/`，非 from-scratch |
| GRPO from scratch（reasoning ch06/07） | ❌ | `post_training/reasoning/training/reinforcement_learning.py` 为 42 行占位 |
| 蒸馏（reasoning ch08） | ❌ | `post_training/reasoning/training/distillation.py` 为占位 |
| 逻辑推理评测（MATH verifier / MMLU / LLM judge） | 🟡 | `post_training/reasoning/evaluation/` 有部分实现 |

## 4. 缺口优先级（教学价值 × 成本）

1. **MLA 完整实现**（预训练）：DeepSeek 骨架已占位，与 GQA 做 KV cache 显存对比实验，教学价值最高。
2. **GRPO from scratch**（后训练）：reasoning-from-scratch 的核心，纯 PyTorch 实现与现有 unsloth 版对照。
3. **推断时扩展脚手架**（推断）：self-consistency / self-refinement 只需组合现有 generate，成本低。
4. **独立 SWA 层 + KV cache 窗口截断**（预训练×推断交叉）。
5. **beam search + repetition penalty**（推断）：小补全，性价比最高。
6. **ALiBi**（预训练）：极低成本。
7. **MoE aux loss / shared expert**（预训练）。
8. **模型 zoo 扩张**（Mistral → OLMo3 → …）：排在接口统一之后，避免复制出更多 drift。

## 相关

- [代码问题清单](bugs.md)
- [落地路线](roadmap.md)
