> 所属：[TinyLLM Docs](../README.md) › [模型推断](index.md)

# 权重加载

权重 IO 的唯一实现位于 `models/io/`，同时服务预训练、后训练和推断。`inference/loaders/` 仅保留旧 import 的兼容 shim。

| 目录 | 模型 | 内容 |
|---|---|---|
| `models/io/gpt2/` | GPT-2 系列 | HF / safetensors / 官方 TF checkpoint、分类头加载 |
| `models/io/qwen3/` | Qwen3 dense/MoE | 下载、权重映射、加载示例 |
| `models/io/gemma3/` | Gemma3 270M | safetensors 加载（WIP） |
| `models/io/llama/` | Llama 2 / 3 | Meta/HF 权重加载 |

## 使用约定

```python
from models.io.gpt2.load_gpt2_pretrained_weights import model_with_gpt2_weights
```

推断统一入口仍推荐 `inference.load_model()`，不要绕过 registry 重复实现加载流程。GPT-2 的 `hf`、`safetensors`、`openai_tf` source 名由底层加载器直接兼容，不再经过额外映射层。

TensorFlow 仅在官方 TF checkpoint 路径懒加载；其他路径不受影响。真实权重下载需要网络与相应凭证，离线测试使用 tiny 随机权重。

## 相关

- [推断入口](index.md)
- [模型](../40-models/index.md)
- [模块迁移测试](../../tests/unit/test_module_moves.py)
