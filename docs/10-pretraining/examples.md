> 所属：[TinyLLM Docs](../README.md) › [Pre-training](index.md)

# 预训练示例

`pre_training/examples/` 是书本 walkthrough 与 standalone 示例，**不是单元测试**，按主题分三组。

## components/（组件逐步实现）

| 文件 | 用途 |
|---|---|
| `tokenizer_bpe.py` | 词级 SimpleTokenizer 编码-解码往返 vs BPE 子词分词 |
| `attention_walkthrough.py` | 自注意力/因果掩码/MHA 逐步实现 |
| `layernorm.py` | LayerNorm 与 PyTorch 对比 |
| `feed_forward.py` | ReLU/GELU/SiLU FFN |
| `activation_gelu.py` | GELU 精确版 vs tanh 近似版（与 HF 对齐的陷阱之一） |
| `text_embedding.py` | token/position embedding、embedding≈matmul |
| `pos_encoding.py` | 可学习位置嵌入 vs 正弦固定编码 |
| `residual_connections.py` | 残差连接与梯度 |
| `block_assembly.py` | 手动组装 GPT-2 并与 Model 参数量对账、外推 124M |

## training/（训练主线示例）

| 文件 | 用途 |
|---|---|
| `train_loop_template.py` | warmup + cosine + 梯度裁剪模板 |
| `gpt2_model_check.py` | GPT-2 前向/参数量检查 |
| `gpt2_pretrain_demo.py` | 小数据训练演示 |

## analysis/（性能分析）

| 文件 | 用途 |
|---|---|
| `flops_analysis.py` | FLOPs / 性能分析（thop；可复用实现见 [tools/profiling](../../tools/profiling/)） |

## 外部参考区

书本 standalone 脚本与独立语料实验在 `pre_training/references/`（不接入主链路）：

- `gpt2_book/`：LLMs-from-scratch ch05 standalone 训练；
- `pgcorpus/`：Gutenberg 独立预训练（`pgcorpus_pretraining_simple.py`，依赖上游 PyPI 包 `llms_from_scratch`）与数据准备（`prepare_dataset.py`），见 [gutenberg](gutenberg.md)。

## 权重加载

GPT-2 权重加载统一在 `models/io/gpt2/`（`load_gpt2_pretrained_weights.py`、`openai_gpt2_weights_load*.py`、`load_pretrained_weights.py`），详见 [权重 IO](../30-inference/loaders.md)。

## 相关

- [training](training.md)
- [extras（references 布局）](extras.md)
- [测试](../50-development/testing.md)
