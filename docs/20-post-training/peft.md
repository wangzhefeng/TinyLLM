> 所属：[TinyLLM Docs](../README.md) › [Post-training](index.md)

# PEFT 方法地图

自研规范实现位于仓库根 `layers/lora.py`（LLMs-from-scratch 附录 E）；
`post_training/peft/` 是围绕它的实验与外部框架实战参考。

## 自研层（layers/lora.py）

| 能力 | 说明 |
|---|---|
| LoRA 缩放 | 标准 `alpha/rank`（论文/HF peft/书最新版口径；换 rank 无需重调学习率） |
| LoRA dropout | 旁路输入 dropout（默认 0.0） |
| 目标层选择 | `replace_linear_with_lora(..., target_keywords=, exclude_keywords=)` |
| DoRA 简化版 ×2 | 不合并（归一化 LoRA 旁路输出）与合并（m·V 权重分解）——**二者数学不等价**，继承自书附录 E 简化版，docstring 已注明 |
| DoRA 论文版 | `LinearWithDoRACorrect`（对 W+ΔW 按输出神经元做 m·V 分解） |
| 权重合并 | `merge_lora_weights` 支持 LoRA 变体折叠为普通 Linear；DoRA 变体 raise TypeError |
| 测试 | `tests/unit/test_lora.py` 25 项 |

## 目录（post_training/peft/）

| 目录/文件 | 方法 | 状态 |
|---|---|---|
| `model_peft/lora_qlora/` | QLoRA 实战：Orca math、SQL、from-scratch | 可参考（from-scratch 部分断链） |
| `model_peft/adapter_finetuning/` | Adapter + DistilBERT 分类 | 示例（有已知 bug，见下） |
| `model_peft/prompt_tuning/` | Prompt Tuning | ✅ 自研 `SoftPromptGPT`（Lester 2021，冻结基座 + n 个可学习虚拟 token，不走 peft 库；测试 8 项） |
| `model_peft/finetuning_feature_extractor/` | 特征提取微调 | 可参考 |
| `model_peft/knowledge_distillation/` | 蒸馏 | 不可运行（见 cleanup-log） |
| `model_peft/deepseek_lora/` | DeepSeek-R1-Distill-Qwen LoRA | 可参考（评测对比有已知 bug） |
| `model_peft/uniem_finetune/` | 嵌入模型微调 | 超出文本 LLM 主线 |
| `model_peft/unsloth_exps/`、`lit_gpt_exps/` | 外部工具实验 | 参考 |
| `layers/merge_adapte_weights.py` | 权重合并 re-export | shim（文件名历史拼写保留） |
| `run/run_gpt_lora_sft.py` | 自研 LoRA SFT 入口 | 空壳（用自研层请走 `run_gpt_clf_sft.py --finetune_method lora`） |

## 端到端自研 LoRA 实验

```bash
# 分类微调 + 自研 LoRA 层（layers/lora.py，exclude out_head）
python post_training/sft/run/run_gpt_clf_sft.py --finetune_method lora
```

## 入口

```bash
python post_training/peft/run_peft.py   # 教程拼接参考，不可直接运行
```

## 相关

- [SFT / Fine-tuning](finetuning/index.md)
- [LoRA 与 QLoRA](finetuning/lora-qlora.md)
- [distillation.md](distillation.md)

## 已知问题

- [ ] `model_peft/lora_qlora/scratch_exps/lora_dora_from_scratch.py` 四重断链（import、`Exp_Basic` 签名、空 `model_dict`、缺 args 键），整体不可运行。
- [ ] `run_peft.py` 为两段教程拼接（TrainingArguments/Trainer 未导入、数据集未定义、训练 mt0-large 推断 opt-350m）。
- [ ] `deepseek_lora/model_eval.py` 的 `merge_and_unload()` 使对比评测变成模型自比；collator 无 -100 mask。
- [ ] QLoRA 训练组合仍全依赖 bitsandbytes；NF4 **量化原语**已自研（`peft/layers/nf4.py`，量化/反量化往返 + 6 项测试），double quantization 与 4bit+LoRA 组合训练留作下一步。
