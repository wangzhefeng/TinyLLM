# 预训练（Pretraining）

> 所属：[TinyLLM Docs](../README.md)

对应 LLMs-from-scratch ch02–ch05 的主线实现，目标是从零训练一个小型 GPT-2 并跑通数据、训练、评估、生成。

## 分页

| 页面 | 内容 |
|---|---|
| [data.md](data.md) | 数据预处理、tokenizer 选择、数据集 |
| [training.md](training.md) | 训练循环、实验类、DP/DDP、超参 |
| [examples.md](examples.md) | 书本 walkthrough、GPT-2 示例 |
| [gutenberg.md](gutenberg.md) | Gutenberg 语料准备 |
| [extras.md](extras.md) | nanotron 参考、legacy 入口 |

## 代码入口

```bash
python run_gpt2_124M.py --help
bash pre_training/scripts/pretrain_gpt2_124M.sh
bash pre_training/scripts/pretrain_gpt2_124M_DP.sh
bash pre_training/scripts/pretrain_gpt2_124M_DDP.sh
```

## 代码结构

```text
pre_training/
├── data/            # 语料加载、Gutenberg
├── experiments/     # 实验类、超参搜索
├── examples/        # 书本示例（非测试）
├── analysis/        # FLOPs 分析
├── references/      # nanotron 参考
├── legacy/          # 失修入口
└── scripts/         # 启动脚本
```

## 相关

- [模型架构](../40-models/architecture.md)
- [数据集说明](../50-development/datasets.md)
- [测试](../50-development/testing.md)
- [LLMs-from-scratch 分析](../60-reference/llms-from-scratch/index.md)
