> 所属：[TinyLLM Docs](../../README.md) › [Post-training](../index.md)

# SFT / Fine-tuning

后训练中的监督微调（SFT）与分类微调。PEFT 单独见 [PEFT](../peft.md)。

## 内容

- [微调数据](data.md)：常用数据、数据格式、数据预处理
- [概念与误区](concepts.md)：Post-training、SFT、LoRA/QLoRA 选择、常见问题
- [最佳实践](practices.md)：模型/方法/数据选择、参数、训练与评估
- [过拟合与欠拟合](overfitting.md)：诊断与处理
- [LoRA 与 QLoRA](lora-qlora.md)：技术原理与实战入口
- [Llama 模型下载](llama-download.md)

## 代码入口

```bash
python post_training/references/trl_sft/run_sft.py --help  # TRL 教程参考（外部栈对照区）
python post_training/sft/run/run_gpt_instruction_sft.py
python post_training/sft/run/run_gpt_clf_sft.py
python post_training/sft/bert_sentiment/run.py --help   # BERT 情感分类微调
```

## 说明

- 代码位于 `post_training/sft/`；数据加载按任务分置 `sft/data_provider/{instruction_follow,text_clf}/`。
- `post_training/sft/bert_sentiment/`：情感分类微调，复用从零 BERT 的预训练 checkpoint（支持 pretrained/scratch/freeze 对照），详见 [BERT](../../40-models/bert.md)。
- 运行前设置 `PYTHONPATH="$PWD"`。

## 2026-10-21 更新

- `run_gpt_clf_sft.py` 支持 `--finetune_method last_layer|last_transformer_stage|all_layers|lora`
  （对应书 ch06 §6.5 冻结策略实验与附录 E LoRA；自研层见 [PEFT](../peft.md)）。
- 指令 SFT 的 prompt-masking collate 已实现
  （`instruction_follow/data_loader.py::custom_collate_fn_mask_out_instruction`，
  对应书 ch07 exercise；`create_dataloader(..., mask_out_instruction=True)` 开启）。
