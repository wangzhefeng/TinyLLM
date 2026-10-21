# 后训练（Post-training）

> 所属：[TinyLLM Docs](../README.md)

覆盖 LLMs-from-scratch ch06–ch07（分类、指令微调、DPO、PEFT）与 reasoning-from-scratch。

## 分页

| 页面 | 内容 |
|---|---|
| [finetuning/index.md](finetuning/index.md) | SFT 数据、概念、最佳实践、LoRA/QLoRA、Llama 下载 |
| [dpo.md](dpo.md) | Direct Preference Optimization |
| [rlhf.md](rlhf.md) | REINFORCE / PPO / DPO / GRPO |
| [distillation.md](distillation.md) | 知识蒸馏（logit 蒸馏与推理轨迹蒸馏） |
| [reasoning.md](reasoning.md) | reasoning-from-scratch 与归属说明 |
| [peft.md](peft.md) | PEFT 方法地图 |

## 代码结构

```text
post_training/
├── sft/          # 分类微调、指令 SFT、评测
│   ├── data_provider/  # text_clf、instruction_follow（含 prompt-masking collate）
│   ├── exp/            # clf、instruction、evaluate
│   ├── layers/         # clf_head（last_layer/last_transformer_stage/all_layers/lora 四策略）
│   ├── run/            # 三个 SFT 入口
│   └── run_sft.py      # TRL/SFTTrainer 参考实现
├── peft/         # LoRA/QLoRA/adapter/prefix/prompt/蒸馏实验
│   ├── model_peft/     # 外部框架（peft/trl/unsloth）实战参考
│   ├── layers/         # lora_dora（shim）、merge_adapter
│   ├── data_provider/  # adapter
│   ├── cfgs/、scripts/
│   └── run_peft.py     # 教程参考（不可直接运行）
├── preference/   # DPO / GRPO / PPO（PPO 为空壳）
│   ├── data_provider/dpo/
│   ├── exp/
│   └── model_rlhf/
├── reasoning/    # training（GRPO/REINFORCE/蒸馏）、inference、evaluation
└── common/       # 共享 data_loader/instruction_format/exp_basic
```

规范 LoRA/DoRA 层实现位于仓库根 `layers/lora.py`（供任意域即插即用）。

## 入口

```bash
# SFT（分类微调：--finetune_method last_layer|last_transformer_stage|all_layers|lora）
python post_training/sft/run/run_gpt_clf_sft.py
# SFT（指令微调）
python post_training/sft/run/run_gpt_instruction_sft.py
# SFT（Ollama 评委打分）
python post_training/sft/run/run_gpt_instruction_sft_evaluate.py

# PEFT（外部框架教程参考，需按文档自查依赖）
python post_training/peft/run_peft.py

# 偏好（DPO：logprob 默认序列求和，--dpo_avg_logprobs 1 切换逐 token 平均）
python post_training/preference/model_rlhf/dpo/run_gpt_dpo.py
```

以上 gpt 入口均支持全部默认参数启动（不再 `required=True`）；数据下载/模型缓存见
[datasets](../50-development/datasets.md)。

## 说明

- 运行前确保仓库根在 `PYTHONPATH`：`export PYTHONPATH="$PWD"`。
- 2026-10-21 已修复历史 stale import（`utils.llm.*`/`models.gpt`）与三个 run 入口崩溃，
  详见 [cleanup-log](../50-development/cleanup-log.md)。
- `reasoning/` 中训练部分属于后训练；推断时扩展与评测更接近推断域，见 [reasoning.md](reasoning.md)。

## 相关

- [模型推断](../30-inference/index.md)
- [测试](../50-development/testing.md)
- [清理记录](../50-development/cleanup-log.md)
