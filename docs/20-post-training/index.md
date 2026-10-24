# 后训练（Post-training）

> 所属：[TinyLLM Docs](../README.md)

覆盖分类/指令 SFT、PEFT、DPO、PPO/GRPO/REINFORCE、蒸馏与 reasoning 训练。

## 分页

| 页面 | 内容 |
|---|---|
| [finetuning](finetuning/index.md) | SFT 数据、概念与实践 |
| [dpo.md](dpo.md) | Direct Preference Optimization |
| [rlhf.md](rlhf.md) | REINFORCE / PPO / DPO / GRPO |
| [distillation.md](distillation.md) | logit 与推理轨迹蒸馏 |
| [reasoning.md](reasoning.md) | reasoning 训练与推断域归属 |
| [peft.md](peft.md) | PEFT 方法地图 |

## 代码结构

```text
post_training/
├── sft/          # 分类/指令 SFT、评测、BERT 情感分类
├── peft/         # LoRA/QLoRA/adapter/prompt 与外部框架参考
├── preference/   # DPO、PPO、偏好数据
├── reasoning/    # training：REINFORCE、蒸馏、reasoning SFT
├── rl/           # 共享采样、logprob、mask、clipped objective
└── common/       # 少量共享数据工具
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
python post_training/preference/model_rlhf/dpo/run_gpt_dpo.py

# TRL 教程参考（非统一主链路）
python post_training/sft/run_sft.py --help
```

默认数据根为 `dataset/finetuning/`。DPO 输入使用 SFT checkpoint，输出使用 `-dpo.pth`，不再覆盖 `-sft.pth` 命名。`peft/run_peft.py` 是不可直接运行的教程拼接，不属于公开主入口。

## 相关

- [模型推断](../30-inference/index.md)
- [测试](../50-development/testing.md)
- [清理记录](../50-development/cleanup-log.md)
