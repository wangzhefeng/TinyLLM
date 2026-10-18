> 所属：[TinyLLM Docs](../README.md) › [Development](index.md)

# 测试

本目录只放**可重复运行的单元测试/集成测试**，不再放教学 demo。

## 运行

```bash
# 全量
python -m unittest discover -s tests -v

# 只跑单元测试
python -m unittest discover -s tests/unit -v
```

## 目录约定

| 目录 | 内容 |
|---|---|
| `unit/` | 单模块、无网络、秒级完成：attention、FFN、norm、tokenizer、GPT 前向/反向、BERT MLM/NSP、BERT 训练流水线（run_bert）、BERT 情感分类（bert_sentiment）、tools（profiling/checkpoint/data） |
| `integration/` | 跨模块或需要小模型生成的测试（后续补充） |

## 说明

- 使用标准库 `unittest`，不额外引入 pytest。
- 测试使用极小配置（embed_dim=32、2 层、vocab=128），CPU/MPS 均可跑。
- 原来的 `test/` 里大部分是书本 walkthrough / FLOPs 分析 / 权重加载 demo，不是单元测试；它们已迁到：
  - `pre_training/examples/`：attention、FFN、LayerNorm、embedding、residual、训练模板、GPT-2 示例；
  - `pre_training/analysis/`：FLOPs 分析；
  - `inference/loaders/gpt2/`：GPT-2 权重加载；
  - `pre_training/examples/gpt2_book/`：书本 ch05 的 standalone 训练示例。
