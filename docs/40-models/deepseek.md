> 所属：[TinyLLM Docs](../README.md) › [Models](index.md)

# DeepSeek（独立模型目录）

本目录用于 DeepSeek V2/V3/R1 的架构学习与从零实现。当前只建立骨架，不包含可运行的完整模型。

## 目录

```text
models/deepseek/
├── model.py       # DeepSeek 模型组装占位（DeepSeekModel）
├── mla.py         # Multi-Head Latent Attention 占位
├── __init__.py
└── README.md
```

## 当前项目中与 DeepSeek 相关的内容

| 位置 | 内容 | 是否归入本目录 | 说明 |
|---|---|---|---|
| `models/deepseek/model.py` | 原 `todo/other/architectures/deepseek/model.py` 的空壳 | ✅ 已迁入 | 改为带说明的 `DeepSeekModel` 占位 |
| `layers/attention.py::MultiHeadLatentAttention` | MLA 空类 | ✅ 已迁入 `mla.py` | 从共享 attention 中移出，避免 DeepSeek 专属逻辑污染通用层 |
| `layers/moe.py` | SparseMoE / MoE FeedForward，注释引用 DeepSeek MoE | ❌ 保留共享 | DeepSeekMoE 的 shared/routed expert 可在此基础上扩展，但通用 MoE 仍服务 Qwen3 等模型 |
| `post_training/peft/model_peft/deepseek_lora/` | DeepSeek-R1-Distill-Qwen-7B 的 LoRA SFT / 评测 | ❌ 留在后训练 | 它实际是 Qwen2 架构的蒸馏模型，不是 DeepSeek V3 架构 |
| `docs/00-overview/architecture.md`、`README.md` | DeepSeek 架构索引/链接 | ❌ 文档 | 仅引用 |
| `docs/30-inference/serving.md` | 提到 DeepSeek-V3 作为推理对象 | ❌ 文档 | 无代码 |

## 尚未覆盖、需要补的内容

DeepSeek V3/R1 的完整实现至少包含：

1. **MLA**：Q/K/V 低秩压缩、decoupled RoPE、只缓存 latent。
2. **DeepSeekMoE**：1 个 shared expert + 多个 routed experts，top-k 路由与负载均衡。
3. **Multi-Token Prediction（MTP）**：训练时的多 token 预测头（V3 引入）。
4. **RMSNorm + SwiGLU + RoPE**：可与 `layers/` 现有组件复用。
5. **权重加载与配置**：V3/R1 的 HuggingFace 权重映射、tokenizer。
6. **推理**：MLA 的 KV cache 与 `inference/` 的 serving 逻辑。

## 实现顺序建议

1. 先用小配置实现 MLA，并与 `layers/attention.py` 的 GQA/MHA 做数值/显存对比。
2. 复用 `layers/moe.py` 的 router，扩展 shared expert 与 top-k 负载均衡。
3. 组装 `DeepSeekModel`，跑通随机输入的前向/反向。
4. 再接权重加载与生成，最后补 MTP 训练。

## 参考

- DeepSeek-V2: <https://arxiv.org/abs/2405.04434>
- DeepSeek-V3: <https://arxiv.org/abs/2412.19437>
- DeepSeek-R1: <https://arxiv.org/abs/2501.12948>
- DeepSeek 官方仓库: <https://github.com/deepseek-ai/DeepSeek-V3>
