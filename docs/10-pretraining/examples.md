> 所属：[TinyLLM Docs](../README.md) › [Pre-training](index.md)

# 预训练示例

> 所属：[TinyLLM Docs](../README.md) › [Pre-training](index.md)

`pre_training/examples/` 是书本 walkthrough 与 standalone 示例，**不是单元测试**。

| 文件 | 用途 |
|---|---|
| `attention_walkthrough.py` | 自注意力/因果掩码/MHA 逐步实现 |
| `feed_forward.py` | ReLU/GELU/SiLU FFN |
| `layernorm.py` | LayerNorm 与 PyTorch 对比 |
| `text_embedding.py` | token/position embedding、embedding≈matmul |
| `residual_connections.py` | 残差连接与梯度 |
| `train_loop_template.py` | warmup + cosine + 梯度裁剪模板 |
| `gpt2_model_check.py` | GPT-2 前向/参数量检查 |
| `gpt2_pretrain_demo.py` | 小数据训练演示 |
| `gpt2_book/` | LLMs-from-scratch ch05 standalone 训练 |
| `flops_analysis.py` | FLOPs / 性能分析（thop；可复用实现见 [tools/profiling](../../tools/profiling/)） |
| `pgcorpus_pretraining_simple.py` | Gutenberg 独立预训练示例（依赖上游 PyPI 包 `llms_from_scratch`） |

## 权重加载

GPT-2 权重加载已统一到 [inference/loaders/gpt2/](../../inference/loaders/gpt2/)：

- `load_gpt2_pretrained_weights.py`
- `openai_gpt2_weights_load*.py`
- `load_pretrained_weights.py`

## 相关

- [training](training.md)
- [测试](../50-development/testing.md)
- [inference/loaders](../30-inference/loaders.md)
