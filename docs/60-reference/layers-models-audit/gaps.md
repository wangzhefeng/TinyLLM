# 缺失主题清单（对照两本上游书）

> 所属：[TinyLLM Docs](../../README.md) › [参考与分析](../index.md) › [layers/models 审查](index.md)

按预训练 / 推断 / 后训练三个视角，列出 `layers/` 与 `models/` 相对
[LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch) 与
[reasoning-from-scratch](https://github.com/rasbt/reasoning-from-scratch) 缺失的主题。
✅ = 已具备；❌ = 缺失；🟡 = 有零件但未闭环。

> **历史快照说明（2026-10-25 追加）**：本清单是 2026-10-20 审查快照，文中
> `post_training/reasoning/{inference,evaluation}/`、`preference/model_rlhf/grpo/`、
> `training/reinforcement_learning.py` 等为历史路径，现已迁移到
> `inference/strategies/`、`evaluation/` 公共库、`post_training/rl/`、
> `post_training/references/trl_grpo/`（见 [cleanup-log](../../50-development/cleanup-log.md)）。

## 1. 预训练视角

### 1.1 注意力与位置编码（layers/）

| 主题 | 状态 | 说明 |
|---|---|---|
| MHA / GQA / SDPA / FlexAttention | ✅ | `layers/attention.py` 8 种实现 |
| MLA（DeepSeek 低秩 latent KV） | ✅ | `models/deepseek/mla.py` 完整实现（2026-10-20）：低秩 latent + decoupled RoPE + latent-only cache，增量=整段一致性有测试 |
| 独立 SWA 层 | ✅ | `layers/attention.py:SlidingWindowAttention`（2026-10-20），含窗口 KV cache；mask 工具在 `layers/masking.py:create_sliding_window_mask` |
| 跨层 KV 共享 | ❌ | KV cache 不支持跨层复用 |
| Gated DeltaNet / DSA | ❌ | 线性注意力与稀疏注意力均缺 |
| RoPE 长上下文扩展 | ✅ | `precompute_rope_params` 新增 NTK-aware（ntk_factor）与 YaRN（yarn_config + mscale），测试见 `test_rope_scaling.py` |
| ALiBi | ✅ | `layers/position_encoding/ALiBi.py`（斜率公式与论文一致，支持非二次幂头数插值） |
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
| Llama3.1 / 3.2 | ✅ | 复用 Llama3 Model + 各自 config（含 rope_freq）；注意 Llama3.2 官方 tying 未实现 |
| DeepSeek-V2/V3（MLA + MoE + MTP） | 🟡 | MLA + DeepSeekModel 已实现（2026-10-20）；DeepSeekMoE 的 shared expert 已在 `layers/moe.py` 支持，MTP 未实现 |
| Mistral / Mixtral | ❌ | SWA 最简形态 + top-2 MoE 教科书案例 |
| OLMo3 / Tiny Aya / Qwen3.5 / Gemma4 / GPT-OSS | ❌ | 上游 ch05 bonus 模型 |
| T5（encoder-decoder） | ❌ | 已有 encoder-only（BERT）与 decoder-only，缺中间形态 |

## 2. 推断视角

| 主题 | 状态 | 说明 |
|---|---|---|
| greedy / temperature / top-k / top-p | ✅ | `inference/generate_core.py:sample_next_token` |
| beam search | ✅ | `inference/generate_core.py:beam_search_generate`（beam 冻结、长度归一化），beam_size=1 与 greedy 一致有测试 |
| repetition / frequency / presence penalty、min-p | 🟡 | repetition penalty 已实现（CTRL 式）；frequency/presence、min-p 未实现 |
| KV cache（朴素 + 预分配） | ✅ | 两种实现，但接口互不兼容 |
| batched 生成 + 逐序列 early stop | ✅ | `generate_text_basic_batched_*_stop` |
| streaming | ✅ | yield 式 |
| 推断时扩展（reasoning ch04/05） | ✅ | `post_training/reasoning/inference/`：self-consistency 投票、Best-of-N、self-refinement、stable 早停（2026-10-20） |
| KV cache × SWA 窗口截断 / ×MLA latent | ✅ | `trim_kv_cache_window`（tuple cache 截断）+ MLA latent-only cache（2026-10-20） |
| 量化（int8/FP8/GGUF）、投机解码 | ❌ | 无 |
| paged KV cache | 🟡 | `inference/serving` 的 nano-vllm 与 layers/ 的 cache 是两套无关联实现 |

## 3. 后训练视角

| 主题 | 状态 | 说明 |
|---|---|---|
| LoRA / DoRA | ✅ | 规范实现迁移至 `layers/lora.py`（新增 merge_lora_weights），`post_training/peft/layers/lora_dora.py` 为兼容 shim；models/ 可即插即用（有测试） |
| 分类头（ch06） | 🟡 | `post_training/sft/layers/clf_head.py` 存在，但 models/ 各模型无 `num_classes` 输出模式 |
| DPO | ✅ | `post_training/preference/exp/exp_finetune_gpt_dpo.py` |
| GRPO（unsloth/vllm 版） | 🟡 | `post_training/preference/model_rlhf/grpo/`，非 from-scratch |
| GRPO from scratch（reasoning ch06/07） | ✅ | `post_training/reasoning/training/reinforcement_learning.py` 纯 PyTorch 实现（组内相对优势 + PPO clip + 可选 KL），学习信号有测试 |
| 蒸馏（reasoning ch08） | ✅ | `distillation.py`：Hinton KL 蒸馏（温度软标签 + alpha 混合硬标签 CE），学生逼近教师有测试 |
| 逻辑推理评测（MATH verifier / MMLU / LLM judge） | 🟡 | `post_training/reasoning/evaluation/` 有部分实现 |

## 4. 缺口优先级（教学价值 × 成本）

> 2026-10-20 更新：第 1–7 项已全部落地（见 roadmap.md Phase 2 与各 ✅ 行）。

1. ✅ **MLA 完整实现**（预训练）
2. ✅ **GRPO from scratch**（后训练）
3. ✅ **推断时扩展脚手架**（推断）
4. ✅ **独立 SWA 层 + KV cache 窗口截断**（预训练×推断交叉）
5. ✅ **beam search + repetition penalty**（推断）
6. ✅ **ALiBi**（预训练）
7. ✅ **MoE aux loss / shared expert**（预训练）
8. **模型 zoo 扩张**（Mistral → OLMo3 → …）：排在接口统一之后，避免复制出更多 drift。

## 相关

- [代码问题清单](bugs.md)
- [落地路线](roadmap.md)
