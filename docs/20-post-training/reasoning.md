> 所属：[TinyLLM Docs](../README.md) › [Post-training](index.md)

# 推理（Reasoning）

reasoning **训练**留在后训练；推断时扩展与通用评测已迁入推断域。旧路径保留兼容 shim，新代码应使用下表的新位置。

## 后训练实现

| 位置 | 内容 |
|---|---|
| `post_training/rl/core.py` | GRPO 采样、组内优势、logprob、completion mask、更新流程 |
| `post_training/rl/objectives.py` | GRPO/PPO 共用 clipped policy objective 与可选 k3 KL |
| `post_training/reasoning/training/reinforce.py` | REINFORCE 与 baseline |
| `post_training/reasoning/training/distillation.py` | logit 蒸馏 |
| `post_training/reasoning/training/distill_reasoning.py` | 教师采样、verifier 过滤、学生 SFT |
| `post_training/preference/model_rlhf/ppo/` | 序列级 bandit PPO + critic |

`reinforcement_learning.py` 旧路径仅 re-export `post_training.rl.core`。PPO 不再反向依赖 reasoning 的 GRPO 模块。

## 推断域实现

| 位置 | 内容 |
|---|---|
| `inference/strategies/voting_and_verification.py` | self-consistency、Best-of-N、嵌套 boxed 解析 |
| `inference/strategies/advance_text_generation.py` | self-refinement 与 stable-answer 早停 |
| `inference/strategies/llm_basic/` | Qwen3 示例 |
| `inference/evaluation/benchmarks/` | MATH-500、MMLU |
| `inference/evaluation/judges/` | LLM judge、Elo、Bradley-Terry |

## 说明

- logit 蒸馏与推理轨迹蒸馏是两条不同链路，分别对应 Hinton 软标签和 rejection-sampling SFT。
- 核心采样、verifier、judge 与排行榜均可离线测试；真实模型和数据在调用时加载。
- 测试：`test_grpo.py`、`test_ppo.py`、`test_reinforce.py`、`test_distillation.py`、`test_inference_time_scaling.py`、`test_llm_judges.py`、`test_leaderboards.py`。
