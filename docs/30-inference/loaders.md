> 所属：[TinyLLM Docs](../README.md) › [模型推断](index.md)

# 权重加载

> 所属：[TinyLLM Docs](../README.md) › [模型推断](index.md)

权重加载统一在 `inference/loaders/`。

| 目录 | 模型 | 内容 |
|---|---|---|
| `gpt2/` | GPT-2 124M/355M/774M/1558M | PyTorch / HF / safetensors / 官方 TF checkpoint 加载、分类头加载 |
| `qwen3/` | Qwen3 0.6B / 30B-A3B | dense/MoE、KV cache demo、下载工具 |
| `gemma3/` | Gemma3 270M | safetensors 加载（WIP：仅下载与加载函数，模型组装未接） |
| `llama/` | Llama 2 / 3 | Meta/HF 权重加载 |

## 说明

- GPT-2 加载器从 `pre_training/examples/gpt2_weight_loading/` 迁入，作为统一入口。
- `post_training/sft/exp/`、`post_training/preference/exp/` 引用 `inference.loaders.gpt2.*`；
  SFT 评测脚本引用 `inference.evaluation.ollama_evaluate`（2026-10-20 已修复断链）。
- `openai_gpt2_weights_load.py` 的 tensorflow 为懒加载，未安装不影响 HF/safetensors 路径。

## 相关

- [serving](serving.md)
- [examples](../10-pretraining/examples.md)
- [模型](../40-models/index.md)
