> 所属：[TinyLLM Docs](../../README.md) › [LLMs-from-scratch 分析](index.md)

> 所属：[TinyLLM Docs](../../README.md) › [LLMs-from-scratch 分析](index.md)

## 4. TinyLLM 现状盘点

### 4.1 项目结构

| 目录 | 角色 | 规模（Python LOC，近似） |
|---|---|---:|
| `layers/` | 可复用层：attention、FFN、norm、RoPE、MoE、KV cache、tokenizer、mask | 7.8k |
| `models/` | 模型组装：GPT-2、Llama、Qwen3、Gemma3、BERT、DeepSeek | 3.0k |
| `exp/` | 实验类：预训练、超参搜索 | 0.9k |
| `utils/` | 训练工具：optimizer、LR、loss、logging、device、checkpoint | 1.7k |
| `data_provider/` | 数据下载/加载、Gutenberg 预处理 | 0.8k |
| `tools/distributed/` | DP/DDP/FSDP/TP/DeepSpeed | 2.8k |
| `apps/` | Chainlit 对话界面 | 0.2k |
| `qwen3_reasoning/` | 推理模型评测/训练占位 | 1.3k |
| `test/minimind/` | MiniMind 参考实现：模型、pretrain、数据、LoRA | 1.9k |
| `test/llmbox/` | SFT 数据适配层 | 0.6k |
| `karpathy_proj/` | minGPT/nanoGPT/nanochat（含 Muon） | 2.9k |

### 4.2 真正可运行的主干

已通过导入检查（`.venv/bin/python`）：

```
layers.attention, layers.kv_cache, layers.moe, layers.inference
models.gpt2.gpt2_124M, models.llama.llama2, models.llama.llama3_8B
models.qwen3.qwen3, models.gemma3.Gemma3_270M
exp.exp_pretrain_gpt2_124M, data_provider.data_loader
utils.optimizer, utils.learning_rate
```

`run_gpt2_124M.py` + `exp/exp_pretrain_gpt2_124M.py` 是与主框架集成度最高的“预训练闭环”；此外还有 `test/smolGPT/`、`test/minimind/`、`models/llama/TinyLlama/pretrain.py` 等参考实现，但它们各自独立、未接入 `exp/`。

### 4.3 占位与失修清单（补内容前建议先修）

| 位置 | 问题 |
|---|---|
| `run_llama.py` / `run_qwen.py` | 调用 `Exp(args)`，但 `Model_Pretrain.__init__` 需要 `(args, global_rank, world_size, local_rank)` → 运行即报错 |
| `run_gemma.py` | 空壳，只有 `main(): pass` |
| `exp/exp_pretrain_llama2_7B.py` 等 5 个 exp | 均为 32 行空壳，只有 `main(): pass` |
| `models/llama/llama31_8B.py`、`llama32_1B.py` | 空壳 |
| `models/deepseek/model.py` | 空壳；`MultiHeadLatentAttention` 也是 `pass` |
| `test/llama_load/*` | 引用 `models.gpt2_model_cfg.*`、`models.llama_load.*` 等重构前路径，已不可运行 |
| `test/gpt2_model/*` 多处 | 仍引用 `models.gpt2_model_cfg.model_cfgs`（当前应为 `config/gpt2/model_cfgs.py`） |
| `scripts/find_near_duplicates.sh` | 指向不存在的 `data_provider/finetune/instruction_follow/find_near_duplicates.py` |
| `qwen3_reasoning/` | `distillation.py`、`reinforcement_learning.py`、`advance_text_generation.py`、`voting_and_verification.py`、`llm_judges*.py` 等均为 42 行空壳 |
| `exp/exp_pretrain_gpt2_124M.py` | warmup、梯度裁剪仍是注释 TODO；LR 使用 epoch 级 `adjust_learning_rate`，不是 token 级 cosine |
| `apps/` | 仅预训练模型对话，没有分类/指令微调 UI |

### 4.4 超出原书的部分

- **分布式训练**：DDP、FSDP（自动/手动 wrapping、checkpoint、mixed precision）、张量并行、DeepSpeed ZeRO、DataParallel。
- **模型广度**：BERT、Encoder-Decoder Transformer（Annotated Transformer）、DyT、Qwen3 30B-A3B MoE 权重加载、Gemma3 270M。
- **推理优化**：KV cache 的 batched/streaming/optimized 版本、top-k/temperature、EOS 停止。
- **工程化**：配置驱动（`config/`）、实验命名、日志、早停、MPS/CPU 适配、Chainlit。
- **参考实现**：MiniMind（tokenizer 训练、SFT/DPO 数据、LoRA）、LLMBox SFT 数据、karpathy nanochat（Muon）。

这些是项目的“差异化资产”，补缺口时应尽量复用，而不是推倒重来。

---
