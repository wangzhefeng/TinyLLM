> 所属：[TinyLLM Docs](../README.md) › [Pre-training](index.md)

# 数据预处理

> 所属：[TinyLLM Docs](../README.md) › [Pre-training](index.md)

本文是数据策略笔记；具体数据目录与下载方式见 [datasets](../50-development/datasets.md)。

## 数据形式

- instruction-response pairs
- 预训练语料：纯文本 / 文档级 JSONL

## 策略

- **Filtering**：质量过滤、去重、长度/语言筛选
- **Synthetic Data**：合成指令数据、instruction pretraining 数据
- **Mixing**：预训练/指令/领域数据的配比
- **Long-context / continued pretraining**：长上下文与继续预训练阶段

## 合成数据参考

- [Magpie](https://arxiv.org/abs/2406.08464)：从对齐 LLM 生成指令数据
- [Instruction Pre-Training](https://arxiv.org/abs/2406.14491)：从原始语料合成指令数据
- [Genie](https://arxiv.org/abs/2401.14367)：content-grounded 数据生成
- [HotpotQA](https://arxiv.org/abs/1809.09600)

## 代码位置

- 语料加载：`pre_training/data/data_loader.py`
- 数据准备：`pre_training/data/`
- tokenizer：`layers/tokenizers/`（见 [tokenizers](../40-models/tokenizers.md)）

## 相关

- [datasets](../50-development/datasets.md)
- [training](training.md)
- [Gutenberg](gutenberg.md)
