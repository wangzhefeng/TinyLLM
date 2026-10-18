> 所属：[TinyLLM Docs](../../README.md) › [LLMs-from-scratch 分析](index.md)

## 5. 覆盖度矩阵（53 个内容单元）

> 状态依据第 1.2 节定义；“证据”列给出项目中的对应位置。

### 5.1 主线与 ch02–ch04

| # | 仓库内容 | TinyLLM 对应 | 状态 | 说明 |
|---:|---|---|:--:|---|
| 1 | ch02 主线：分词、滑动窗口数据集、DataLoader | `layers/tokenizers/`、`data_provider/data_loader.py` | ✅ | 多种 tokenizer + 滑动窗口 |
| 2 | ch02/05 BPE from scratch | `layers/tokenizers/bpe/simple_bpe.py` | ✅ | 含 train/encode/decode、加载 GPT-2 vocab |
| 3 | ch02/02+03+04 bonus | `test/text_embedding_test.py`、`data_loader.py` | ✅ | 概念均有对应 |
| 4 | ch02/06 SimpleTokenizerV3 | `layers/tokenizers/simple_custom.py` | 🟡 | 有简单 tokenizer，但无“空白往返” |
| 5 | ch03 主线：自注意力/MHA | `layers/attention.py` | ✅ | 实现齐全 |
| 6 | ch03/02 高效 MHA | `layers/attention.py` | ✅ | SDPA、FlexAttention、einsum 等 |
| 7 | ch03/03 buffers | `SharedBuffers` | ✅ | 缓存 mask/RoPE |
| 8 | ch04 主线：GPT 模型 | `models/gpt2/gpt2_124M.py` | ✅ | 124M 配置 + 生成 |
| 9 | ch04/02 性能分析 | `test/flops_analysis.py`、`utils/model_memory.py` | ✅ | thop + 显存估算 |
| 10 | ch04/03 KV cache | `layers/kv_cache.py`、`layers/inference.py` | ✅ | naive + optimized + batched/streaming |
| 11 | ch04/04 GQA | `layers/attention.py` | ✅ | 含 Qwen3/Gemma3 变体 |
| 12 | ch04/05 MLA | `MultiHeadLatentAttention` | ❌ | 空类；DeepSeek 模型空壳 |
| 13 | ch04/06 SWA | Gemma3 local mask | 🟡 | 只有模型内 mask，无独立实现/显存估算 |
| 14 | ch04/07 MoE | `layers/moe.py`、Qwen3 MoE | ✅ | SparseMoE + MoE FFN |
| 15 | ch04/08 Gated DeltaNet | 无 | ❌ | 无实现 |
| 16 | ch04/09 DSA | 无 | ❌ | 无实现 |
| 17 | ch04/10 跨层 KV 共享 | 无 | ❌ | 无实现 |

### 5.2 ch05 预训练与模型 zoo

| # | 仓库内容 | TinyLLM 对应 | 状态 | 说明 |
|---:|---|---|:--:|---|
| 18 | ch05 主线预训练闭环 | `exp/exp_pretrain_gpt2_124M.py` | ✅ | 含 AMP/DDP/早停/采样 |
| 19 | ch05/02 替代权重加载 | `test/gpt2_model/gpt2_load/` | 🟡 | 代码存在但路径 stale |
| 20 | ch05/03 Gutenberg | `data_provider/pgcorpus_gutenberg/` | ✅ | 数据准备 + 预训练脚本 |
| 21 | ch05/04 LR schedulers | `utils/learning_rate.py`、`test/train_loop_template.py` | 🟡 | 函数有，主循环未接入 |
| 22 | ch05/05 超参搜索 | `exp/hparam_search.py` | ✅ | 网格搜索 |
| 23 | ch05/06 用户界面 | `apps/app_orig.py`、`app_own.py` | ✅ | Chainlit 替代 Streamlit |
| 24 | ch05/07 GPT→Llama | `models/llama/`、`test/llama_load/` | 🟡 | 模型有，加载/入口失修 |
| 25 | ch05/08 内存高效加载 | 无 | ❌ | 无分块 load_state_dict |
| 26 | ch05/09 tokenizer 扩展 | 无 | ❌ | 无特殊 token + embedding resize |
| 27 | ch05/10 训练加速 | AMP/compile/SDPA/DDP/fused AdamW | 🟡 | 缺 vocab padding、梯度累积、基准 |
| 28 | ch05/11 Qwen3 | `models/qwen3/`、`test/qwen3_model/` | ✅ | 0.6B + 30B-A3B + KV cache |
| 29 | ch05/12 Gemma3 | `models/gemma3/` | ✅ | 270M + 权重加载 |
| 30 | ch05/13 OLMo3 | 无 | ❌ | 7B/32B 均无 |
| 31 | ch05/14 其他 LLM 跑 ch05 | `run_llama.py`、`run_qwen.py` | 🟡 | 入口签名不匹配，跑不通 |
| 32 | ch05/15 Tiny Aya | 无 | ❌ | 无 |
| 33 | ch05/16 Qwen3.5 | 无 | ❌ | 无线性注意力混合 |
| 34 | ch05/17 Gemma4 | 无 | ❌ | 无 KV 共享/SWA 组合 |
| 35 | ch05/18 Muon | `karpathy_proj/nanochat/muon.py` | 🟡 | 有实现，未接入 optimizer/训练 |
| 36 | ch05/19 CPU/MPS 差异 | — | ➖ | 硬件说明，无需代码 |

### 5.3 ch06–ch07 微调与对齐

| # | 仓库内容 | TinyLLM 对应 | 状态 | 说明 |
|---:|---|---|:--:|---|
| 37 | ch06 主线分类微调 | `models/bert/`、`calc_loss` 的 `clf_sft` 分支 | ❌ | 无 GPT 分类闭环 |
| 38 | ch06/02 19 组对照实验 | 无 | ❌ | 无 |
| 39 | ch06/03 IMDb 对比 | 无（仅 THUCNews） | ❌ | 无 GPT/BERT baseline 对比 |
| 40 | ch06/04 分类 UI | 无 | ❌ | 现有 UI 只做生成 |
| 41 | ch07 主线指令微调 | `test/llmbox` 数据类、`test/minimind` 数据类 | ❌ | 有数据，无 SFT 训练 |
| 42 | ch07/02 数据集工具 | 无（脚本悬空） | ❌ | 近重复/语态构造缺失 |
| 43 | ch07/03 模型评测 | math verifier + multiple choice | 🟡 | judge 空壳、leaderboard 不完整 |
| 44 | ch07/04 DPO | 仅 `DPODataset` | ❌ | 无 DPO loss/训练 |
| 45 | ch07/05 合成数据 | 无 | ❌ | 无 Ollama/OpenAI/reflection |
| 46 | ch07/06 指令 UI | 无 | ❌ | 无 |

### 5.4 附录、工程化与推理书

| # | 仓库内容 | TinyLLM 对应 | 状态 | 说明 |
|---:|---|---|:--:|---|
| 47 | 附录 A：PyTorch + DDP | `tools/distributed/` | ✅ | 且扩展到 FSDP/TP/DeepSpeed |
| 48 | 附录 B/C：文献/习题 | — | ➖ | 文档性质 |
| 49 | 附录 D：warmup/cosine/clip | `test/train_loop_template.py`、smolGPT | 🟡 | 有样板，未进主循环 |
| 50 | 附录 E：LoRA | `test/minimind/model/model_lora.py` | 🟡 | 仅 MiniMind，无 GPT-2 通用 LoRA |
| 51 | `pkg/llms_from_scratch` | `layers/`、`models/` + `pyproject.toml` | 🟡 | 无稳定 API/测试/CI |
| 52 | `setup/` 环境说明 | `README.md` + `uv` | ✅ | 已覆盖 |
| 53 | `reasoning-from-scratch`（8 章） | `qwen3_reasoning/` | ❌ | 13 个模块中 9 个为空壳 |

### 5.5 分领域主观评分

| 领域 | 覆盖率（估） | 判断依据 |
|---|---:|---|
| 预训练主线 | 85% | 数据、模型、训练、生成、权重加载齐全 |
| 架构模块 | 65% | MHA/GQA/MoE/KV 强，MLA/SWA/DeltaNet/DSA/KV sharing 弱 |
| 微调与对齐 | 10% | ch06/ch07/DPO/LoRA 基本空白 |
| 训练工程 | 55% | 分布式强，scheduler/加速/内存优化未闭环 |
| 数据工程 | 40% | 预训练数据强，指令/偏好数据只有适配器 |
| 评测 | 25% | 数学/选择题有，LLM judge/leaderboard 空 |
| 工程化 | 45% | 结构清晰，但入口失修、无 CI、无公共 API |
| 推理模型 | 10% | 目录齐全，实现多为占位 |

---
