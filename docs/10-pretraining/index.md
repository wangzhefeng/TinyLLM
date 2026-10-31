# 预训练（Pretraining）

> 所属：[TinyLLM Docs](../README.md)

对应 LLMs-from-scratch ch02–ch05 的主线实现，目标是从零训练一个小型 GPT-2 并跑通数据、训练、评估、生成。

## 分页

| 页面 | 内容 |
|---|---|
| [data.md](data.md) | 数据预处理、tokenizer 选择、数据集 |
| [training.md](training.md) | 训练循环、实验类、DP/DDP、超参 |
| [examples.md](examples.md) | 书本 walkthrough、GPT-2 示例 |
| [ablation.md](ablation.md) | 组件升级消融实验（RMSNorm/SwiGLU/RoPE/GQA） |
| [gutenberg.md](gutenberg.md) | Gutenberg 语料准备 |
| [extras.md](extras.md) | nanotron 参考等扩展 |

## 代码入口

```bash
python run_gpt2_124M.py --help
python run_bert.py --help            # BERT（MLM + NSP），见 [BERT](../40-models/bert.md)
python run_llama.py --help           # Llama2 / Qwen3 入口（tiny 演示配置）
python run_qwen.py --help
bash pre_training/scripts/pretrain_gpt2_124M.sh
bash pre_training/scripts/pretrain_gpt2_124M_DP.sh
bash pre_training/scripts/pretrain_gpt2_124M_DDP.sh
```

## 代码结构

```text
pre_training/
├── data_provider/   # data_loader.py 语料加载；bert/ 数据层（DAPT TSV）
├── experiments/     # exp_basic.py（经 models/registry 解析模型）、GPT-2 预训练实验、exp_component_ablation.py、bert/dapt_data.py
├── examples/        # 教材示例三分组（非测试）
│   ├── components/  # tokenizer / attention / layernorm / FFN / GELU / embedding / 位置编码 / 残差 / block 组装
│   ├── training/    # 预训练 demo、训练循环模板、模型检查
│   └── analysis/    # FLOPs 分析
├── references/      # 外部参考（单向离线）：gpt2_book/、pgcorpus/、nanotron/
└── scripts/         # 启动脚本
```

## 相关

- [模型架构](../40-models/architecture.md)
- [数据集说明](../50-development/datasets.md)
- [测试](../50-development/testing.md)
- [LLMs-from-scratch 分析](../60-reference/llms-from-scratch/index.md)
