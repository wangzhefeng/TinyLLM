# 后训练（Post-training）

> 所属：[TinyLLM Docs](../README.md)

覆盖 LLMs-from-scratch ch06–ch07（分类、指令微调、DPO、PEFT）与 reasoning-from-scratch。

## 分页

| 页面 | 内容 |
|---|---|
| [finetuning/index.md](finetuning/index.md) | SFT 数据、概念、最佳实践、LoRA/QLoRA、Llama 下载 |
| [dpo.md](dpo.md) | Direct Preference Optimization |
| [rlhf.md](rlhf.md) | PPO / DPO / GRPO |
| [distillation.md](distillation.md) | 知识蒸馏 |
| [reasoning.md](reasoning.md) | reasoning-from-scratch 与归属说明 |
| [peft.md](peft.md) | PEFT 方法地图 |

## 代码结构

```text
post_training/
├── sft/          # 分类微调、指令 SFT、评测
│   ├── data_provider/  # text_clf、instruction_follow、llmbox
│   ├── exp/            # clf、instruction、evaluate
│   ├── layers/         # clf_head
│   ├── run/            # 三个 SFT 入口
│   ├── scripts/
│   └── run_sft.py
├── peft/         # LoRA/QLoRA/adapter/prefix/prompt/蒸馏实验
│   ├── model_peft/
│   ├── layers/         # lora_dora、merge_adapter
│   ├── data_provider/  # adapter
│   ├── cfgs/、scripts/
│   └── run_peft.py
├── preference/   # DPO / GRPO / PPO
│   ├── data_provider/dpo/
│   ├── exp/
│   ├── model_rlhf/
│   └── scripts/
├── reasoning/    # training、inference、evaluation
└── common/       # 共享 data_loader/instruction_format/exp_basic
```

## 入口

```bash
# SFT
python post_training/sft/run_sft.py --help
python post_training/sft/run/run_gpt_instruction_sft.py
python post_training/sft/run/run_gpt_clf_sft.py

# PEFT
python post_training/peft/run_peft.py

# 偏好
python post_training/preference/model_rlhf/dpo/run_gpt_dpo.py
```

## 说明

- 运行前确保仓库根在 `PYTHONPATH`：`export PYTHONPATH="$PWD"`。
- 旧脚本中仍存在 `utils.llm`、`models.gpt` 等历史依赖缺失，属于原有失修代码，不是本次移动引入。
- `reasoning/` 中训练部分属于后训练；推理时扩展与评测更接近推理域，见 [reasoning.md](reasoning.md)。

## 相关

- [推理域](../30-inference/index.md)
- [测试](../50-development/testing.md)
- [清理记录](../50-development/cleanup-log.md)
