> 所属：[TinyLLM Docs](../README.md) › [Post-training](index.md)

# 推理（Reasoning）

对应 [reasoning-from-scratch](https://github.com/rasbt/reasoning-from-scratch) 的实现目录。

> **归属说明**：reasoning **训练**（RLVR/GRPO、REINFORCE、蒸馏、reasoning SFT）属于后训练；
> **推断时扩展**（CoT、self-consistency、best-of-n、self-refinement）和 **推理评测**更接近推断域，
> 目前为了对照书本内容暂放在本目录，后续可拆分到 `inference/`。

## 目录

| 目录 | 内容 | 状态 |
|---|---|---|
| `training/reinforcement_learning.py` | GRPO from scratch：组内相对优势 + PPO clip + 可选 k3 KL + completion mask + 梯度裁剪 + 多 prompt 批处理（`grpo_update_epoch`） | ✅ 可用（2026-10-21 增强） |
| `training/reinforce.py` | REINFORCE / REINFORCE-with-baseline（ch06 教学链第一环，GRPO 对照组） | ✅ 可用（2026-10-21 新增） |
| `training/distillation.py` | Hinton logit 蒸馏（温度软标签 + 可选硬标签混合） | ✅ 可用 |
| `training/distill_reasoning.py` | ch08 推理轨迹蒸馏：教师采样 → verifier 过滤 → 学生 SFT（rejection sampling 蒸馏） | ✅ 可用（2026-10-21 新增） |
| `inference/voting_and_verification.py` | self-consistency 投票（嵌套 \boxed 解析）、Best-of-N | ✅ 可用（2026-10-21 增强） |
| `inference/advance_text_generation.py` | self-refinement 循环、stable-answer 早停 | ✅ 可用 |
| `inference/llm_basic/` | Qwen3 demo（书本 checkpoint 经 remap 加载） | 可用 |
| `evaluation/benchmark_based_eval/` | MATH-500 答案解析与评分、MMLU 多项选择（Method 1 生成式 + Method 2 first-token logprob + 评测循环 + 随机基线） | ✅ 可用（2026-10-21 增强，import 零副作用） |
| `evaluation/judgement_based_eval/` | LLM judge（Ollama/Qwen3 后端）、Elo、Bradley-Terry（Adam 拟合 + 400/log10 缩放） | ✅ 可用（2026-10-21 补全） |

## 说明

- 评测脚本运行时才下载数据/模型（`load_math500_data()` 惰性加载），import 无副作用、可单测。
- `inference/llm_basic/qwen3_06B.py` 在导入时就会加载模型，适合作为脚本运行，不宜直接 import 到单元测试。
- 蒸馏两条线的区别：`distillation.py` 是 **logit 蒸馏**（教师软标签，Hinton 2015）；
  `distill_reasoning.py` 是 **推理轨迹蒸馏**（教师生成 → 验证过滤 → 学生 SFT，ch08 主线）。
- 单测：`tests/unit/test_grpo.py`、`test_reinforce.py`（含轨迹蒸馏）、`test_distillation.py`、
  `test_inference_time_scaling.py`、`test_leaderboards.py`、`test_llm_judges.py`。
