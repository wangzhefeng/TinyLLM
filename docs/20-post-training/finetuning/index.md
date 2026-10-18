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
python post_training/sft/run_sft.py --help
python post_training/sft/run/run_gpt_instruction_sft.py
python post_training/sft/run/run_gpt_clf_sft.py
```

## 说明

- 代码位于 `post_training/sft/`；共享数据加载在 `post_training/common/data_provider/`。
- 运行前设置 `PYTHONPATH="$PWD"`。
