> 所属：[TinyLLM Docs](../README.md) › [Post-training](index.md)

# Reinforcement Learning with Human Feedback(RLHF)

按教学链顺序（reasoning-from-scratch ch06）：

| 方法 | 实现 | 状态 |
|---|---|---|
| REINFORCE / with-baseline | `post_training/reasoning/training/reinforce.py` | ✅ from scratch |
| GRPO（RLVR 主线） | `post_training/rl/`（core + objectives + batched + grpo_variants） | ✅ from scratch（completion mask + clip-higher + k1/k3 KL + Dr.GRPO 归一 + 批量采样；真实训练入口 `--task grpo`，见 [reasoning](reasoning.md)） |
| DPO | `post_training/preference/`（自研）+ `model_rlhf/grpo/*`（TRL 参考） | ✅ 自研可运行 |
| PPO | `post_training/preference/model_rlhf/ppo/main.py` | ✅ from scratch（序列级 bandit：critic 价值基线 + clip + 可选 k3 KL；与 GRPO 对照——GRPO 用组归一化优势替代 critic） |
| Reward Model（RLHF 一阶段） | `post_training/preference/rm/`（2026-10-24） | ✅ from scratch（价值头 + BT 成对损失） |

## DPO 要点（2026-10-21 修复后）

- policy/reference 均从 **SFT checkpoint** 出发（先 HF 初始化再载 checkpoint，顺序不可颠倒）；
- logprob 默认**序列求和**（论文公式 7 口径）；`--dpo_avg_logprobs 1` 切换逐 token 平均做对比实验；
- 损失：`-logsigmoid(beta · ((π_c - π_r) - (π_ref_c - π_ref_r)))`，跟踪 chosen/rejected reward margins。

## PPO 要点（2026-10-21 新增）

- `ppo_update_step(model, optimizer, prompt_ids, reward_fn, ...)`：actor 与 critic 分开反传/step；
- 优势 A = r - V(s)（critic 为 nn.Linear(embed_dim, 1)，输入 prompt 末 token embedding，detach 防梯度污染）；
- 也可用 `value_fn=` 注入外部 baseline（与 critic 互斥）。
- 教学差异：同玩具任务下 PPO+常数基线的优势幅度（±0.5）弱于 GRPO 组归一化（±2），
  梯度信号更弱——对照实验见 tests/unit/test_ppo.py。

## RM 要点（2026-10-24 新增，RLHF 一阶段）

- `RewardModel`（`post_training/preference/rm/rm_head.py`）：causal LM backbone +
  `v_head = Linear(embed_dim, 1)` 逐 token 标量 reward；序列得分取最后一个
  非 pad token 位置（右 padding 约定，与 DPO collate 一致）；
- `bradley_terry_loss`：`-log σ(r_chosen - r_rejected)`；`rm_pairwise_step`
  封装单步成对训练（前向/反传/更新，返回 loss 与 reward margin）；
- 数据复用 DPO 偏好对（`data_provider/dpo`）；训练编排见
  `exp/exp_train_rm.py`；GPT-2 `forward` 新增 `output_hidden_states`
  选项暴露 final hidden（默认行为不变）；
- 训练后的 RM 可作为 PPO（`model_rlhf/ppo/main.py`）的 learned reward，
  补全 SFT → RM → PPO 的完整 RLHF 三阶段。

## GRPO 参考

- [Qwen Scheduler GRPO](https://huggingface.co/anakin87/qwen-scheduler-7b-grpo)
- DeepSeekMath/GRPO: https://arxiv.org/abs/2402.03300
- ch07 改进变体（DeepSeek V3.2 style / OLMo3 style（DAPO·Dr.GRPO）/ GDPO）移植于
  `post_training/rl/grpo_variants.py`，各函数 docstring 注明原文路径；
- 教学链对照：REINFORCE → REINFORCE-with-baseline → GRPO（组归一化优势 + PPO clip），
  见 [reasoning](reasoning.md)。
