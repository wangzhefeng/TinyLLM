> 所属：[TinyLLM Docs](../README.md) › [Post-training](index.md)

# 推理（Reasoning）

对应 [reasoning-from-scratch](https://github.com/rasbt/reasoning-from-scratch) 的实现目录。

> **归属说明**：reasoning **训练**（RLVR/GRPO、蒸馏、reasoning SFT）属于后训练；  
> **推断时扩展**（CoT、self-consistency、best-of-n、self-refinement）和 **推理评测**更接近推断域，  
> 目前为了对照书本内容暂放在本目录，后续可拆分到 `inference/`。

## 目录

| 目录 | 内容 | 状态 |
|---|---|---|
| `evaluation/benchmark_based_eval/` | MATH-500 答案解析、选择题评测 | 部分可用 |
| `evaluation/judgement_based_eval/` | LLM judge、Elo/Bradley-Terry、leaderboard | 多为占位 |
| `training/reinforcement_learning.py` | GRPO from scratch（组内相对优势 + PPO clip + 可选 KL） | ✅ 可用（2026-10-20） |
| `training/distillation.py` | KL 蒸馏（温度软标签 + 可选硬标签混合） | ✅ 可用（2026-10-20） |
| `inference/voting_and_verification.py` | self-consistency 投票、Best-of-N、\\boxed 答案抽取 | ✅ 可用（2026-10-20） |
| `inference/advance_text_generation.py` | self-refinement 循环、stable-answer 早停 | ✅ 可用（2026-10-20） |
| `inference/llm_basic/` | Qwen3 demo（书本 checkpoint 经 remap 加载） | 可用 |

## 说明

- 评测脚本会下载/读取 MATH-500，需要 Hugging Face 数据集缓存与网络权限。
- `inference/llm_basic/qwen3_06B.py` 在导入时就会加载模型，适合作为脚本运行，不宜直接 import 到单元测试。
- 推断时扩展（CoT、self-consistency 等）目前留在本目录；训练/推断/评测已分别放在 `training/`、`inference/`、`evaluation/`。
- 单测：`tests/unit/test_grpo.py`、`test_distillation.py`、`test_inference_time_scaling.py`。
