# 后训练（Post-training）

> 所属：[TinyLLM Docs](../README.md)

覆盖分类/指令 SFT、PEFT、DPO、RM、PPO/GRPO/REINFORCE、蒸馏与 reasoning 训练。

## 分页

| 页面 | 内容 |
|---|---|
| [finetuning](finetuning/index.md) | SFT 数据、概念与实践 |
| [dpo.md](dpo.md) | Direct Preference Optimization |
| [rlhf.md](rlhf.md) | REINFORCE / PPO / DPO / GRPO / RM |
| [distillation.md](distillation.md) | logit 与推理轨迹蒸馏 |
| [reasoning.md](reasoning.md) | reasoning 训练与 rl/references 边界 |
| [peft.md](peft.md) | PEFT 方法地图与外部栈 references |

## 代码结构

```text
post_training/
├── run.py        # 统一分发：sft_clf / sft_ins / dpo / grpo / reinforce / distill_reasoning
├── sft/          # 分类/指令 SFT、evaluation/judge.py（alpaca 打分）、BERT 情感分类
├── peft/         # demos/（lora_demo、prompt_tuning、scratch_exps）、layers/merge_adapter_weights.py
├── preference/   # DPO（run/run_gpt_dpo.py）、RM（rm/）、偏好数据
├── reasoning/    # 蒸馏训练、教师数据生成（data_gen.py）
├── rl/           # GRPO/PPO/REINFORCE 共享实现与训练入口
└── references/   # 外部栈对照实验区：trl_sft/、trl_grpo/、hf_peft/、unsloth/
```

预训练、分类 SFT、指令 SFT、DPO 已共用 `training.Trainer` 与统一 checkpoint。规范 LoRA/DoRA 位于 `layers/lora.py`。

## 入口

```bash
# 统一入口
python post_training/run.py --task sft_clf --help
python post_training/run.py --task sft_ins --help
python post_training/run.py --task dpo --help

# 原入口仍保留
python post_training/sft/run/run_gpt_clf_sft.py
python post_training/sft/run/run_gpt_instruction_sft.py
python post_training/preference/run/run_gpt_dpo.py
```

默认数据根为 `dataset/finetuning/`。DPO 输入使用 SFT checkpoint，输出使用 `-dpo.pth`，不再覆盖 `-sft.pth` 命名。

## references/ 的定位

`post_training/references/` 收纳**外部栈对照实验**（TRL SFT/GRPO、bitsandbytes、Unsloth、HF peft 教程），

- 保持上游原貌，便于对照自研实现；
- **不接入 `run.py` 分发器**，不是可运行主链路；
- 各子目录依赖与数据自行管理，见 [peft](peft.md) 与 [rlhf](rlhf.md)。

## 相关

- [模型推断](../30-inference/index.md)
- [评测公共库](../35-evaluation/index.md)
- [测试](../50-development/testing.md)
- [清理记录](../50-development/cleanup-log.md)
