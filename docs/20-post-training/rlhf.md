> 所属：[TinyLLM Docs](../README.md) › [Post-training](index.md)

# Reinforcement Learning with Human Feedback(RLHF)

按教学链顺序（reasoning-from-scratch ch06）：

| 方法 | 实现 | 状态 |
|---|---|---|
| REINFORCE / with-baseline | `post_training/reasoning/training/reinforce.py` | ✅ from scratch |
| GRPO（RLVR 主线） | `post_training/reasoning/training/reinforcement_learning.py` | ✅ from scratch（completion mask + clip + 可选 KL + 多 prompt 批处理） |
| DPO | `post_training/preference/`（自研）+ `model_rlhf/grpo/*`（TRL 参考） | ✅ 自研可运行 |
| PPO | `post_training/preference/model_rlhf/ppo/` | ❌ 空壳，未实现 |

## DPO 要点（2026-10-21 修复后）

- policy/reference 均从 **SFT checkpoint** 出发（先 HF 初始化再载 checkpoint，顺序不可颠倒）；
- logprob 默认**序列求和**（论文公式 7 口径）；`--dpo_avg_logprobs 1` 切换逐 token 平均做对比实验；
- 损失：`-logsigmoid(beta · ((π_c - π_r) - (π_ref_c - π_ref_r)))`，跟踪 chosen/rejected reward margins。

## GRPO 参考

- [Qwen Scheduler GRPO](https://huggingface.co/anakin87/qwen-scheduler-7b-grpo)
- DeepSeekMath/GRPO: https://arxiv.org/abs/2402.03300
- 教学链对照：REINFORCE → REINFORCE-with-baseline → GRPO（组归一化优势 + PPO clip），
  见 [reasoning](reasoning.md)。
