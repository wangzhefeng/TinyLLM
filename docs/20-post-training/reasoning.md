> 所属：[TinyLLM Docs](../README.md) › [Post-training](index.md)

# 推理（Reasoning）

reasoning **训练**留在后训练；推断时扩展策略在推断域，评测基础设施在公共库
[evaluation](../35-evaluation/index.md)。RL 训练原子集中在 `post_training/rl/`。

## 后训练实现

| 位置 | 内容 |
|---|---|
| `post_training/rl/core.py` | GRPO 采样、组内优势、logprob、completion mask、更新流程 |
| `post_training/rl/objectives.py` | GRPO/PPO 共用 clipped policy objective：clip-higher、k1/k3 KL、token_mean/seq_sum_mean、训练诊断 |
| `post_training/rl/batched.py` | 批量组采样与批量单步更新（ch06 batched 工作流，greedy 下与串行版逐位一致） |
| `post_training/rl/rewards.py` | RLVR reward：答案正确性（verifier 懒加载）与 `<think>`/`\boxed{}` 格式奖励 |
| `post_training/rl/tracking.py` | MetricsTracker 指标收集/CSV/摘要、clip fraction、matplotlib 绘图 |
| `post_training/rl/grpo_variants.py` | ch07 改进变体：DeepSeek V3.2 style、OLMo3 style（DAPO/Dr.GRPO）、GDPO |
| `post_training/rl/reinforce.py` | REINFORCE 与 baseline（原 `reasoning/training/reinforce.py`，解开 rl↔reasoning 环） |
| `post_training/rl/ppo.py` | 序列级 bandit PPO + critic（原 `preference/model_rlhf/ppo/main.py`） |
| `post_training/rl/run.py` | 真实训练入口：`python post_training/run.py --task grpo`（或 `--task reinforce`） |
| `post_training/reasoning/training/distillation.py` | logit 蒸馏 |
| `post_training/reasoning/training/distill_reasoning.py` | 教师采样、verifier 过滤、学生 SFT |
| `post_training/references/trl_grpo/` | TRL GRPO 对照实验（gsm8k/smollm/scheduler，不接入分发器） |

旧 `reinforcement_learning.py` 垫片与 `reasoning/{inference,evaluation}` 别名目录已删除，直接使用上表路径。

## 推断域实现

| 位置 | 内容 |
|---|---|
| `inference/strategies/voting_and_verification.py` | self-consistency、Best-of-N、嵌套 boxed 解析 |
| `inference/strategies/advance_text_generation.py` | self-refinement 与 stable-answer 早停 |
| `inference/harness/math500.py` | MATH-500 端到端评测循环 |
| `evaluation/verifiers/`、`evaluation/judges/` | MATH 验证、LLM judge、Elo、Bradley-Terry（见 [评测公共库](../35-evaluation/index.md)） |

## 说明

- logit 蒸馏与推理轨迹蒸馏是两条不同链路，分别对应 Hinton 软标签和 rejection-sampling SFT。
- 核心采样、verifier、judge 与排行榜均可离线测试；真实模型和数据在调用时加载。
- GRPO 真实训练：`python post_training/run.py --task grpo --model qwen3-0.6b --train-size 50`（REINFORCE 对照：`--task reinforce`）；指标写 `logs/grpo/*.csv`，checkpoint 每 `--save-every` 步存 `results/checkpoints/grpo/`。
- 测试：`test_grpo.py`、`test_ppo.py`、`test_reinforce.py`、`test_distillation.py`、`test_inference_time_scaling.py`、`test_llm_judges.py`、`test_leaderboards.py`。
