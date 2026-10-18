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

- 语料加载：`pre_training/data_provider/data_loader.py`
- 数据准备：`pre_training/data_provider/`
- tokenizer：`layers/tokenizers/`（见 [tokenizers](../40-models/tokenizers.md)）

## 相关

- [datasets](../50-development/datasets.md)
- [training](training.md)
- [Gutenberg](gutenberg.md)

## 待办（2026-10-18 盘点）

- [ ] **多语料混合配比**：Filtering/Mixing/Synthetic 目前只是上方策略笔记，无实现
- [ ] **bin/memmap 预 tokenize**：`LLMDataset` 仍是整文本 `tokenizer.encode()` 进内存，Gutenberg 全量级别不可用
- [ ] **文档级 packing / EOS 分隔**：`tools/data/packing.py` 有雏形，未接入预训练链路
