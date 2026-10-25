> 所属：[TinyLLM Docs](../README.md) › [Post-training](index.md)

# PEFT 方法地图

自研规范实现位于仓库根 `layers/lora.py`（LLMs-from-scratch 附录 E）；
`post_training/peft/` 是围绕它的实验与 demo；`post_training/references/` 是外部栈对照实验区。

## 自研层（layers/lora.py）

| 能力 | 说明 |
|---|---|
| LoRA 缩放 | 标准 `alpha/rank`（论文/HF peft/书最新版口径；换 rank 无需重调学习率） |
| LoRA dropout | 旁路输入 dropout（默认 0.0） |
| 目标层选择 | `replace_linear_with_lora(..., target_keywords=, exclude_keywords=)` |
| DoRA 简化版 ×2 | 不合并（归一化 LoRA 旁路输出）与合并（m·V 权重分解）——**二者数学不等价**，继承自书附录 E 简化版，docstring 已注明 |
| DoRA 论文版 | `LinearWithDoRACorrect`（对 W+ΔW 按输出神经元做 m·V 分解） |
| 权重合并 | `merge_lora_weights` 支持 LoRA 变体折叠为普通 Linear；DoRA 变体 raise TypeError |
| NF4 量化原语 | `layers/quantization/nf4.py`：量化/反量化、double quantization（scale 组内 uint8 二次量化 ≈4.127 bit/参数，测试 9 项） |
| 测试 | `tests/unit/test_lora.py`、`test_nf4.py`、`test_prompt_tuning.py` |

## 目录（post_training/peft/）

| 目录/文件 | 内容 | 状态 |
|---|---|---|
| `demos/lora_demo.py` | LoRA 挂载/合并最小演示 | ✅ 自研层示例 |
| `demos/prompt_tuning.py` | `SoftPromptGPT`（Lester 2021，冻结基座 + n 个可学习虚拟 token，不走 peft 库） | ✅（测试 8 项） |
| `demos/scratch_exps/` | 从零手写实验：`lora_dora_from_scratch.py`、`MLP.py`、`exp_basic.py` | 参考（from-scratch 版有已知断链，见下） |
| `layers/merge_adapter_weights.py` | adapter 权重合并工具（原 `merge_adapte_weights.py` 已改名） | ✅ |
| `cfgs/`、`data_provider/adapter/`、`scripts/` | QLoRA 配置与 spectrum 训练脚本 | 配套 |

## 端到端自研 LoRA 实验

```bash
# 分类微调 + 自研 LoRA 层（layers/lora.py，exclude out_head）
python post_training/sft/run/run_gpt_clf_sft.py --finetune_method lora
```

## 外部栈对照区（post_training/references/）

保持上游原貌，**不接入 `run.py` 分发器**；需要各自依赖与数据，不属于默认可运行入口。

| 目录 | 上游栈 | 内容 |
|---|---|---|
| `hf_peft/` | HF peft / Transformers | `run_peft.py` 教程拼接、`adapter_finetuning/`、`deepseek_lora/`、`finetuning_feature_extractor/`、`knowledge_distillation/`、`lit_gpt_exps/`、`uniem_finetune/` |
| `trl_sft/` | TRL SFT | `run_sft.py`、`sql_create_context/`、`orca_math/` QLoRA 实战 |
| `trl_grpo/` | TRL GRPO | `gsm8k/`、`smollm/`、`scheduler/`、`bitsandbytes.py` 4bit 量化对照 |
| `unsloth/` | Unsloth | Qwen3-14B 对话微调、`env_setup/`、`fourbit_models_list.py` |

## 已知问题（随代码迁移，路径为现位置）

- [ ] `references/hf_peft/lit_gpt_exps`（原 `scratch_exps` 谱系）的 `lora_dora_from_scratch.py`
  在 `peft/demos/scratch_exps/` 下保留自研副本，四重断链（import、`Exp_Basic` 签名、空 `model_dict`、缺 args 键），整体不可运行。
- [ ] `references/hf_peft/run_peft.py` 为两段教程拼接（TrainingArguments/Trainer 未导入、数据集未定义、训练 mt0-large 推断 opt-350m）。
- [ ] `references/hf_peft/deepseek_lora/model_eval.py` 的 `merge_and_unload()` 使对比评测变成模型自比；collator 无 -100 mask。
- [ ] QLoRA 训练组合仍全依赖 bitsandbytes；NF4 量化原语已自研（见上表），
  4bit 存储 + LoRA 旁路组合训练留作下一步。

## 相关

- [SFT / Fine-tuning](finetuning/index.md)
- [LoRA 与 QLoRA](finetuning/lora-qlora.md)
- [distillation.md](distillation.md)
