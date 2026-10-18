> 所属：[TinyLLM Docs](../../README.md) › [LLMs-from-scratch 分析](index.md)

## 6. 缺口清单与优先级

> 工作量口径：**S** ≈ 0.5–1 天，**M** ≈ 2–3 天，**L** ≈ 1 周以上（按熟悉 PyTorch、能复用现有代码估算）。

### P0：补齐“书的后半本”，形成完整闭环

| 优先级 | 缺口 | 为什么重要 | 建议落点 | 工作量 |
|---|---|---|---|---|
| P0-1 | ch07 指令微调 SFT | 从“会预训练”到“会做助手”的关键；项目已有数据适配器，只差训练 | `data_provider/finetune/instruction/`、`exp/exp_sft.py`、`run_sft.py` | M |
| P0-2 | ch06 GPT 分类微调 | 复用 GPT-2 权重，成本最低；`calc_loss` 已预留 `clf_sft` | `data_provider/classification/`、`exp/exp_classification.py` | S–M |
| P0-3 | 附录 D 接入主循环 | warmup + cosine + grad clip 是训练稳定性的基本功 | 改造 `exp/exp_pretrain_gpt2_124M.py` | S |
| P0-4 | 附录 E 通用 LoRA | 低成本微调，和 SFT/分类都能组合 | `layers/lora.py`、`utils/optimizer.py` 参数组 | S–M |
| P0-5 | ch07 DPO | 偏好对齐的入口；理解 RLHF 的最小实现 | `layers/loss/dpo.py`、`data_provider/finetune/preference/`、`exp/exp_dpo.py` | M |

### P1：补齐 2025–2026 架构演进

| 优先级 | 缺口 | 关键实现点 | 建议落点 | 工作量 |
|---|---|---|---|---|
| P1-1 | MLA | 压缩 K/V latent、cache 只存 latent、推理时反投影 | `layers/attention.py` + `models/deepseek/` | M |
| P1-2 | 独立 SWA + 显存估算 | local mask、KV cache 窗口截断、5:1 混合 | `layers/attention.py`、`layers/kv_cache.py` | S–M |
| P1-3 | 跨层 KV 共享 | 后层复用前层 K/V，减少 cache 层数 | `layers/kv_cache.py` + 模型组装 | S–M |
| P1-4 | Gated DeltaNet | 线性注意力/RNN 状态更新、门控 | `layers/attention.py` 或新文件 | M |
| P1-5 | DSA | Lightning Indexer + top-k mask | `layers/attention.py` | M |
| P1-6 | OLMo3 / Tiny Aya / Qwen3.5 / Gemma4 | 各自的结构差异（post-norm、并行 block、线性注意力混合、KV 共享） | `models/<name>/` | 每个 M |
| P1-7 | Muon 接入 | 2D 参数用 Muon，其余 AdamW；LR 匹配 | `utils/optimizer.py` + `exp/` | S |

### P2：工程化与数据/评测

| 优先级 | 缺口 | 建议落点 | 工作量 |
|---|---|---|---|
| P2-1 | 内存高效权重加载 | `utils/load_save_data.py` | S |
| P2-2 | tokenizer 扩展（特殊 token + embedding resize） | `layers/tokenizers/` | S |
| P2-3 | 近重复检测 / 被动语态构造 | `data_provider/finetune/` + `scripts/` | S |
| P2-4 | LLM-as-judge 评测（Ollama/OpenAI） | `qwen3_reasoning/evaluation/judgement_based_eval/` | M |
| P2-5 | 合成数据生成（Ollama/reflection） | `data_provider/finetune/synthetic/` | M |
| P2-6 | 公共 API + 测试 + CI | `pyproject.toml`、`tests/`、GitHub Actions | M |
| P2-7 | 修入口与 stale import | `run_llama.py`、`run_qwen.py`、`run_gemma.py`、`test/` | S |
| P2-8 | `reasoning-from-scratch` 8 章 | `qwen3_reasoning/` 按章补全 | L |

---

## 7. 建议的落地路线

### Phase 1：把“后半本”跑通（1–2 周）

1. 修复入口与导入路径，保证 `run_llama.py` / `run_qwen.py` / `test/` 可运行。
2. 在 `exp/exp_pretrain_gpt2_124M.py` 接入 warmup + cosine + grad clip（参考 appendix D 与 `test/train_loop_template.py`）。
3. 新增 `run_classification.py`：SMS spam 数据集 + GPT-2 分类头，目标 test acc ≈ 95%。
4. 新增 `run_sft.py`：Alpaca 1100 条，prompt mask 作为可选项，目标 loss 正常下降、生成格式正确。
5. 新增 `layers/lora.py`，在 SFT/分类上验证 LoRA 可训练、可合并。

**验收**：上述命令可复现实验；loss/accuracy 与参考实现同量级；生成结果可读。

### Phase 2：偏好对齐与评测（1 周）

1. DPO：构造偏好数据（可先用 Ollama 生成），实现 DPO loss 与 reference model。
2. 评测：把 `llm_judges*.py` 补成可用的 Ollama/OpenAI judge，输出胜率/评分。
3. 数据集工具：近重复检测、被动语态构造，接到 `scripts/`。

**验收**：DPO 训练后 chosen/rejected 的 reward margin 上升；评测能产出对比表。

### Phase 3：架构补课（2–3 周）

按“性价比”排序：SWA → 跨层 KV 共享 → MLA → Muon → DeltaNet → DSA。
每个实现配一个最小验证：
- KV cache 与无 cache 的 greedy 输出逐 token 一致；
- GQA/MLA/SWA 的显存估算与公式一致；
- 新模型能与 HF 权重对齐（前向 logits 误差在容差内）。

### Phase 4：模型 zoo 与推理书（按兴趣）

- 先补 OLMo3（post-norm + SWA）与 Tiny Aya（并行 block），再做 Qwen3.5（线性注意力混合）与 Gemma4（KV 共享组合）。
- 若继续推理方向，按 `reasoning-from-scratch` 的 Ch3→Ch8 顺序补：评测 → 推理时扩展 → GRPO → 蒸馏。

---

## 8. 工程建议

1. **先修入口，再补内容**：目前多个 `run_*.py` 与 `test/` 脚本不可运行，会让“已覆盖”的判断失真。
2. **统一实验接口**：`Exp_Basic` 的 `__init__(args, local_rank)` 与 `Model_Pretrain.__init__(args, global_rank, world_size, local_rank)` 不一致，建议统一为 `(args, rank_info)` 或让 `run_*` 统一传参。
3. **把“零件”变成“闭环”**：每个缺口至少配一个 `run_*.py` + 一个 `test/` 验证脚本，否则容易再次变成占位。
4. **保留 Apache 2.0 归属**：仓库是 Apache 2.0；项目已在 `test/gpt2_model/local_test/previous_chapters.py` 保留版权头，移植时继续保留。
5. **优先复用现有资产**：SFT 数据用 `test/llmbox`，LoRA/DPO 数据用 `test/minimind`，Muon 用 `karpathy_proj/nanochat`，不要重复造轮子。
6. **控制范围**：不必把 42 个 bonus 全部搬进来；按“主线闭环 → 架构演进 → 推理书”的优先级取舍。

---
