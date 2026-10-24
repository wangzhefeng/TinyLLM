> 所属：[TinyLLM Docs](../README.md) › [Models](index.md)

# DeepSeek（独立模型目录）

DeepSeek V2/V3/R1 架构学习与从零实现。**MLA、模型组装与 MTP 已实现**
（教学小配置版，2026-10-20 / 2026-10-24）。

## 目录

```text
models/deepseek/
├── model.py       # DeepSeekModel / DeepSeekBlock（MLA + RMSNorm + SwiGLU，支持 KV cache）
├── mla.py         # MultiHeadLatentAttention 完整实现（低秩 latent + decoupled RoPE）
├── mtp.py         # MTPHead 多 token 预测头 + deepseek_mtp_loss（V3 §2.5.1，2026-10-24）
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

## MTP 实现要点（`mtp.py`，2026-10-24）

1. `MTPHead`：`RMSNorm(2d)` 归一化 concat[h_prev, embed(x)] → `Linear(2d→d)` 投影 →
   复用 `DeepSeekBlock` → 共享主模型的 `final_norm` / `out_head`（论文共享设计）。
2. `forward(input_ids, h_prev, start_pos, cache)` 返回 `(logits, h_new)`，
   `h_new` 供下一深度 MTP 级联；cache/start_pos 语义与 `DeepSeekBlock` 一致。
3. `DeepSeekModel.attach_mtp(n_mtp)` 构建共享参数的 `ModuleList`，不改变 forward 签名。
4. 测试：`tests/unit/test_mtp.py`（形状、loss 反传、共享对象同一性、参数去重、cache 两段前向）。

## 与 V3 的差距（尚未覆盖）

1. **DeepSeekMoE**：shared expert + routed experts + 无辅助损失负载均衡（扩展点：`layers/moe.py`）。
2. **权重加载与配置**：V3/R1 的 HuggingFace 权重映射、tokenizer。
3. MLA 的 weight-absorption 推断优化与 serving 集成。
4. MTP 与训练循环的完整整合（当前提供模块与损失，训练编排待接线）。

## 当前项目中与 DeepSeek 相关的内容

| 位置 | 内容 | 是否归入本目录 | 说明 |
|---|---|---|---|
| `models/deepseek/model.py` | `DeepSeekModel` 组装 | ✅ 已迁入 | MLA + SwiGLU，支持 KV cache |
| `models/deepseek/mla.py` | MLA 完整实现 | ✅ 已迁入 | 从共享 attention 中移出，避免 DeepSeek 专属逻辑污染通用层 |
| `layers/moe.py` | SparseMoE / MoE FeedForward | ❌ 保留共享 | DeepSeekMoE 可在此基础上扩展，但通用 MoE 仍服务 Qwen3 等模型 |
| `post_training/peft/model_peft/deepseek_lora/` | DeepSeek-R1-Distill-Qwen-7B 的 LoRA SFT / 评测 | ❌ 留在后训练 | 它实际是 Qwen2 架构的蒸馏模型，不是 DeepSeek V3 架构 |

## 实现顺序（已完成 1–4）

1. ✅ 小配置实现 MLA，与 GQA/MHA 做数值/显存对比（`tests/unit/test_mla.py`）。
2. ⬜ DeepSeekMoE：复用 `layers/moe.py`，扩展 shared expert 与 top-k 负载均衡。
3. ✅ 组装 `DeepSeekModel`，随机输入前向/反向 + cache 生成跑通。
4. ✅ MTP 多 token 预测头（`mtp.py`，2026-10-24）。
5. ⬜ 权重加载与生成。

## 参考

- DeepSeek-V2: <https://arxiv.org/abs/2405.04434>
- DeepSeek-V3: <https://arxiv.org/abs/2412.19437>
- DeepSeek-R1: <https://arxiv.org/abs/2501.12948>
- DeepSeek 官方仓库: <https://github.com/deepseek-ai/DeepSeek-V3>
