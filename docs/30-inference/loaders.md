> 所属：[TinyLLM Docs](../README.md) › [Inference](index.md)

# 权重加载

> 所属：[TinyLLM Docs](../README.md) › [Inference](index.md)

权重加载统一在 `inference/loaders/`。

| 目录 | 模型 | 内容 |
|---|---|---|
| `gpt2/` | GPT-2 124M/355M/774M/1558M | PyTorch / HF / safetensors 加载、分类头加载 |
| `qwen3/` | Qwen3 0.6B / 30B-A3B | dense/MoE、KV cache demo |
| `gemma3/` | Gemma3 270M | safetensors 加载 |
| `llama/` | Llama 2 / 3 | Meta/HF 权重加载 |

## 说明

- GPT-2 加载器从 `pre_training/examples/gpt2_weight_loading/` 迁入，作为统一入口。
- `post_training/sft/exp/` 已改为引用 `inference.loaders.gpt2.*`。
- 部分加载器依赖 `tensorflow`/`safetensors`；缺失时只影响对应脚本。

## 相关

- [serving](serving.md)
- [examples](../10-pretraining/examples.md)
- [模型](../40-models/index.md)
