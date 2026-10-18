> 所属：[TinyLLM Docs](../README.md) › [Post-training](index.md)

# PEFT 方法地图

代码位于 `post_training/peft/model_peft/`，包含多种参数高效微调与相关实验。

## 目录

| 目录/文件 | 方法 | 状态 |
|---|---|---|
| `lora_demo.py` | LoRA（PEFT 库） | 示例 |
| `qlora_demo.py` | QLoRA | 空壳（已删） |
| `adalora_demo.py`、`bitfit_demo.py`、`lorafa_demo.py`、`increlora_demo.py` | 其他 PEFT | 空壳（已删） |
| `lora_qlora/` | QLoRA 实战：Orca math、SQL、from-scratch | 可参考 |
| `adapter_finetuning/` | Adapter + DistilBERT 分类 | 示例 |
| `prefix_tuning/`、`prompt_tuning/` | Prefix/Prompt Tuning | prompt 有示例，prefix 已删 |
| `finetuning_all_layers/`、`finetuning_last_layers/`、`finetuning_feature_extractor/` | 全量/部分/特征提取 | 后两者可参考 |
| `knowledge_distillation/` | 蒸馏 | 见 [distillation](distillation.md) |
| `deepseek_lora/` | DeepSeek-R1-Distill-Qwen LoRA | 可参考 |
| `uniem_finetune/` | 嵌入模型微调 | 超出文本 LLM 主线 |
| `unsloth_exps/`、`lit_gpt_exps/` | 外部工具实验 | 参考 |

## 入口

```bash
python post_training/peft/run_peft.py
```

## 相关

- [SFT / Fine-tuning](finetuning/index.md)
- [LoRA 与 QLoRA](finetuning/lora-qlora.md)
- [distillation.md](distillation.md)
