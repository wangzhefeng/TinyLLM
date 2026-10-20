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

## Phase 1：结构整改（约 1 周）

1. `layers/normailzation/` → `layers/normalization/`，全局更新 import；`CasualAttention` → `CausalAttention`（保留别名兼容）。
2. `layers/inference.py` 归位 `inference/` 域，6+ 个 generate 变体收敛为 1 个通用 `generate()`（流式/batch/cache 用参数与回调表达）。
3. 删除全部 `sys.path.append(Path.cwd())` hack，改正式包导入（`from layers.xxx import ...`）。
4. 统一 `BaseModel` 接口：`forward(x, cache=None, mask=None)` + `reset_kv_cache()` + `tie_weights()`；统一 embedding 属性名与 config 访问方式。
5. 收敛 Qwen3 四变体为单文件可选参数（original/optimized/batched 由构造参数切换）。
6. 修剩余 P1/P2：硬编码 bf16 改 `cfg.dtype`、mask 语义统一为 True=屏蔽并加组合函数、`self.training` 覆盖问题、llama `nn.Sequential` 改 `ModuleList`。

**验收**：全部测试绿 + 生成冒烟（GPT-2/Qwen3 加载官方权重各生成 20 token）；repo-map、architecture、40-models 页面同步更新。

## Phase 2：按三域补教学主题

按教学价值 × 成本排序，每项配"实现 + 最小验证 + 文档页"：

| 顺序 | 主题 | 域 | 最小验证 |
|---|---|---|---|
| 1 | MLA 完整实现（DeepSeek） | 预训练 | 与 GQA 做 KV cache 显存对比；前向形状正确 |
| 2 | beam search + repetition penalty | 推断 | 与 greedy 输出对比用例 |
| 3 | ALiBi | 预训练 | 与 RoPE 同配置前向对照 |
| 4 | 独立 SWA 层 + KV cache 窗口截断 | 预训练×推断 | 窗口外 token 不影响输出 |
| 5 | 推断时推理增强（self-consistency / self-refinement / Best-of-N） | 推断×推理 | MATH-500 小子集准确率提升 |
| 6 | MoE aux loss + shared expert | 预训练 | 路由分布可视化、aux loss 下降 |
| 7 | GRPO from scratch（reasoning ch06/07） | 后训练 | GSM8K 小子集 reward 上升 |
| 8 | 蒸馏（reasoning ch08） | 后训练 | 教师-学生 loss 下降 |
| 9 | layers/ 通用 LoRA 层 | 后训练 | 任意 models/ 模型即插即用、可合并 |
| 10 | RoPE NTK / YaRN 外推 | 预训练×推断 | 超长上下文不退化 |
| 11 | 模型 zoo：Mistral → OLMo3 → Qwen3.5 / Gemma4 | 预训练 | 与 HF 权重前向对齐（容差内） |

**完成标准**（沿用 AGENTS.md）：代码可运行 + 相关文档已更新 + 链接可跳转 + 索引已更新。

## 相关

- [缺失主题清单](gaps.md)
- [代码问题清单](bugs.md)
- [旧版路线（llms-from-scratch/gaps.md，已失真）](../llms-from-scratch/gaps.md)
