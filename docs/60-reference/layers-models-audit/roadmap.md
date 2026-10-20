# 落地路线（layers/models 整改）

> 所属：[TinyLLM Docs](../../README.md) › [参考与分析](../index.md) › [layers/models 审查](index.md)

基于 [缺失主题清单](gaps.md) 与 [代码问题清单](bugs.md) 的三阶段路线。
原则：**先修后学**——所有学习都建立在正确代码之上；每步同步更新文档并验证可运行。

## Phase 0：修 P0 + 回归测试（约 2–3 天）✅ 已完成（2026-10-20）

1. ~~修复 [bugs.md](bugs.md) 中全部 P0（#1–#11）~~ ✅
2. ~~同步修低成本 P1：moe 噪声类型与链式赋值（#12/#13）、cache dtype（#17）、RMSNorm fp32 上转（#14）~~ ✅；另修复 `KVCache_optimized.append` 自动分配与溢出检查、`FeedForwardSiLU` 的 d_ff/hidden_dim 回退（兼容 DotDict 的 KeyError 语义）、SwiGLU 空壳。
3. ~~补最小回归测试（`tests/unit/test_layers_regression.py`，16 项）~~ ✅

**验收结果**：`python -m unittest discover -s tests` 83 项全绿；`scripts/check_docs.py` 通过；
冒烟验证 Llama2/3 自带 config 构造、Gemma3 tying、Qwen3 original/optimized cache 生成均正常。

## Phase 1：结构整改（约 1 周）✅ 基本完成（2026-10-20）

1. ~~`layers/normailzation/` → `layers/normalization/`~~ ✅；`CasualAttention` → `CausalAttention`（保留别名）✅
2. ~~`layers/inference.py` 归位 `inference/` 域~~ ✅（`inference/generate_core.py`，17 处 import 更新）。
   generate 变体收敛**部分完成**：全部复用 `sample_next_token`/`_resolve_ctx_len`，并以回归测试防 drift；合并为单函数推迟。
3. ~~删除全部 `sys.path.append(Path.cwd())` hack~~ ✅（layers/ 与 models/ 共 37 处）
4. 接口统一**轻量完成**：embedding 统一命名 `tok_embed`（llama 同步改）、`reset_kv_cache` 语义统一、
   llama `nn.Sequential` → `ModuleList`；完整 `BaseModel` 抽象基类推迟到 Phase 2 需要时。
5. ~~收敛 Qwen3 重复~~ ✅：删除 `qwen3_original.py` 与 `TransformerBlockQwen3_original`，
   书本 checkpoint 经 `remap_book_state_dict` 重映射加载。
6. ~~硬编码 bf16 改 `cfg.dtype`~~ ✅；~~mask 语义统一为 True=屏蔽并加组合函数~~ ✅
   （`layers/masking.py:combine_causal_padding_mask`）；~~`self.training` 覆盖问题~~ ✅；
   ~~llama `nn.Sequential` 改 `ModuleList`~~ ✅；另修 `qwen3_batched.py` 的
   `pos_id`/`pos_ids` 参数名不匹配（审查遗漏的新 bug）与 Qwen3 RoPE 越界防护（#19）。

**验收结果**：85 项测试全绿；fp32 配置的 Qwen3/Llama3/Gemma3 前向全程 fp32；
batched Qwen3 前向/生成、越界防护冒烟通过；repo-map、40-models、cleanup-log 已同步。

## Phase 2：按三域补教学主题 ✅ 主体完成（2026-10-20）

| 顺序 | 主题 | 域 | 状态 |
|---|---|---|---|
| 1 | MLA 完整实现（DeepSeek） | 预训练 | ✅ `models/deepseek/mla.py` + `DeepSeekModel`，增量=整段一致性与显存优势有测试 |
| 2 | beam search + repetition penalty | 推断 | ✅ `inference/generate_core.py`，beam1=greedy 一致 |
| 3 | ALiBi | 预训练 | ✅ `layers/position_encoding/ALiBi.py`，公式/外推/SDPA 集成有测试 |
| 4 | 独立 SWA 层 + KV cache 窗口截断 | 预训练×推断 | ✅ `SlidingWindowAttention` + `trim_kv_cache_window`，窗口性质有测试 |
| 5 | 推断时扩展脚手架 | 推断 | ✅ self-consistency / Best-of-N / self-refinement / stable 早停 |
| 6 | MoE aux loss + shared expert | 预训练 | ✅ `layers/moe.py` Switch 式 aux loss + DeepSeekMoE shared expert |
| 7 | GRPO from scratch（reasoning ch06/07） | 后训练 | ✅ 纯 PyTorch（组内优势 + PPO clip + 可选 KL），学习信号有测试 |
| 8 | 蒸馏（reasoning ch08） | 后训练 | ✅ Hinton KL 蒸馏 + alpha 混合，学生逼近教师有测试 |
| 9 | layers/ 通用 LoRA 层 | 后训练 | ✅ `layers/lora.py` 规范实现（含 merge），peft 目录为兼容 shim |
| 10 | RoPE NTK / YaRN 外推 | 预训练×推断 | ✅ `ntk_factor` / `yarn_config` + `yarn_attention_mscale` |
| 11 | 模型 zoo | 预训练 | ⬜ 未做（llama31/32 空壳已填；Mistral/OLMo3 等后续） |

测试总量：141 项（含新增 58 项），`python -m unittest discover -s tests` 全绿，`check_docs.py` 通过。

**完成标准**（沿用 AGENTS.md）：代码可运行 + 相关文档已更新 + 链接可跳转 + 索引已更新。

## 相关

- [缺失主题清单](gaps.md)
- [代码问题清单](bugs.md)
- [旧版路线（llms-from-scratch/gaps.md，已失真）](../llms-from-scratch/gaps.md)
