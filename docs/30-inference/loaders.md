> 所属：[TinyLLM Docs](../README.md) › [模型推断](index.md)

# 权重 IO（models/io）

权重加载的唯一实现位于 `models/io/`，同时服务预训练、后训练和推断。推断统一入口是
`inference.load_model()`（注册表 `models/registry.py`），不要绕过它重复实现加载流程。

| 目录 | 模型 | 内容 |
|---|---|---|
| `models/io/gpt2/` | GPT-2 系列 | HF / safetensors / 官方 TF checkpoint、分类头加载 |
| `models/io/qwen3/` | Qwen3 dense/MoE | 下载、权重映射、加载示例 |
| `models/io/gemma3/` | Gemma3 270M | safetensors 下载与映射（已闭环，2026-10-24：属性名对齐模型、tying 分支、`_load_gemma3` 接入 registry） |
| `models/io/llama/` | Llama 2 / 3 | Meta/HF 权重加载（Llama 3.2 1B tying 支持） |
| `models/io/phi/` | Phi-1.5 | HF safetensors 下载与 Wqkv 拆分映射（2026-10-24） |
| `models/io/export/` | Qwen3 -> HF | 本地权重导出 HF Transformers 目录（2026-10-24，reasoning ch08） |

## 使用约定

```python
from models.io.gpt2.load_gpt2_pretrained_weights import model_with_gpt2_weights
```

GPT-2 的 `hf`、`safetensors`、`openai_tf` source 名由底层加载器直接兼容。TensorFlow 仅在官方 TF checkpoint 路径懒加载；其他路径不受影响。真实权重下载需要网络与相应凭证，离线测试使用 tiny 随机权重。

## HF 导出（models/io/export）

`export_to_hf(model, out_dir, cfg, tokenizer_dir=None)` 把本地 Qwen3 state_dict 键名
反向映射到 HF Qwen3 命名，safetensors 落 `model.safetensors`，并生成 HF 格式
`config.json` 与 `generation_config.json`。仅支持 dense（MoE 显式拒绝）。

```python
from models.io.export import hf_export

hf_export.export_to_hf(model, "results/hf_export", config, tokenizer_dir="downloaded_models/qwen3_model/Qwen3-0.6B-base-for-reasoning")
```

wrapper 方案 `hf_export.TinyLLMForCausalLM` 继承 `transformers.PreTrainedModel` 包住本地
Model 的 forward，transformers 懒加载（未安装时模块导入不受影响）。参考上游
rasbt/reasoning-from-scratch `ch08/06_use_via_huggingface`。

## 相关

- [推断入口](index.md)
- [模型](../40-models/index.md)
