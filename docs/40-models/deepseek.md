> 所属：[TinyLLM Docs](../README.md) › [Models](index.md)

# DeepSeek（独立模型目录）

DeepSeek V2/V3/R1 架构学习与从零实现。**MLA 与模型组装已实现**（2026-10-20，教学小配置版）。

## 目录

```text
models/deepseek/
├── model.py       # DeepSeekModel / DeepSeekBlock（MLA + RMSNorm + SwiGLU，支持 KV cache）
├── mla.py         # MultiHeadLatentAttention 完整实现（低秩 latent + decoupled RoPE）
└── __init__.py
```

## MLA 实现要点（`mla.py`）

1. **K/V 低秩压缩**：`kv_down_proj` 把 K/V 压到 `kv_lora_rank` 维 latent `c_kv`，
   推断时只缓存 `c_kv + k_rope`（显存与头数解耦，对比 MHA 节省约一个数量级，见 `tests/unit/test_mla.py`）。
2. **Decoupled RoPE**：位置旋转只作用于每头 `qk_rope_head_dim` 子维，nope 子维不受位置约束，
   因此 latent 反投影后仍能正确施加位置信息。
3. **Q 压缩可选**：`q_lora_rank=None` 时不压缩（与论文一致的两档配置）。
4. **不做 weight absorption**：每步从 latent 反投影（教学优先清晰性）；
   推断服务级优化见 DeepSeek 官方仓库。
5. Cache 协议与 Qwen3 一致：`forward(x, cache)`，`KVCache` 每层存 `(c_kv, k_rope)`；
   增量解码与整段前向逐 token 一致（回归测试覆盖）。

## 与 V3 的差距（尚未覆盖）

1. **DeepSeekMoE**：shared expert + routed experts + 无辅助损失负载均衡（扩展点：`layers/moe.py`）。
2. **Multi-Token Prediction（MTP）**：训练时多 token 预测头（V3 引入）。
3. **权重加载与配置**：V3/R1 的 HuggingFace 权重映射、tokenizer。
4. MLA 的 weight-absorption 推断优化与 serving 集成。

## 当前项目中与 DeepSeek 相关的内容

| 位置 | 内容 | 是否归入本目录 | 说明 |
|---|---|---|---|
| `models/deepseek/model.py` | 原 `todo/other/architectures/deepseek/model.py` 的空壳 | ✅ 已迁入 | 改为带说明的 `DeepSeekModel` 占位 |
| `layers/attention.py::MultiHeadLatentAttention` | MLA 空类 | ✅ 已迁入 `mla.py` | 从共享 attention 中移出，避免 DeepSeek 专属逻辑污染通用层 |
| `layers/moe.py` | SparseMoE / MoE FeedForward，注释引用 DeepSeek MoE | ❌ 保留共享 | DeepSeekMoE 的 shared/routed expert 可在此基础上扩展，但通用 MoE 仍服务 Qwen3 等模型 |
| `post_training/peft/model_peft/deepseek_lora/` | DeepSeek-R1-Distill-Qwen-7B 的 LoRA SFT / 评测 | ❌ 留在后训练 | 它实际是 Qwen2 架构的蒸馏模型，不是 DeepSeek V3 架构 |
| `docs/00-overview/architecture.md`、`README.md` | DeepSeek 架构索引/链接 | ❌ 文档 | 仅引用 |
| `docs/30-inference/serving.md` | 提到 DeepSeek-V3 作为推断对象 | ❌ 文档 | 无代码 |

## 尚未覆盖、需要补的内容

DeepSeek V3/R1 的完整实现至少包含：

1. **MLA**：Q/K/V 低秩压缩、decoupled RoPE、只缓存 latent。
2. **DeepSeekMoE**：1 个 shared expert + 多个 routed experts，top-k 路由与负载均衡。
3. **Multi-Token Prediction（MTP）**：训练时的多 token 预测头（V3 引入）。
4. **RMSNorm + SwiGLU + RoPE**：可与 `layers/` 现有组件复用。
5. **权重加载与配置**：V3/R1 的 HuggingFace 权重映射、tokenizer。
6. **推断**：MLA 的 KV cache 与 `inference/` 的 serving 逻辑。

## 实现顺序建议（已完成 1–3）

1. ✅ 小配置实现 MLA，与 GQA/MHA 做数值/显存对比（`tests/unit/test_mla.py`）。
2. ⬜ DeepSeekMoE：复用 `layers/moe.py`，扩展 shared expert 与 top-k 负载均衡。
3. ✅ 组装 `DeepSeekModel`，随机输入前向/反向 + cache 生成跑通。
4. ⬜ 权重加载与生成，最后补 MTP 训练。

## 参考

- DeepSeek-V2: <https://arxiv.org/abs/2405.04434>
- DeepSeek-V3: <https://arxiv.org/abs/2412.19437>
- DeepSeek-R1: <https://arxiv.org/abs/2501.12948>
- DeepSeek 官方仓库: <https://github.com/deepseek-ai/DeepSeek-V3>
