> 所属：[TinyLLM Docs](../README.md) › [Overview](index.md)

# TinyLLM 架构与目录边界

> 更新时间：2026-10-18（第三次整理：三个域内部按职责再分层）  
> 目标：**预训练 → 后训练 / reasoning → 推理** 三域分离，域内按“数据 / 实验 / 入口 / 扩展”分层。

## 1. 顶层结构

```text
TinyLLM/
├── pre_training/          # 预训练
├── post_training/         # 后训练：SFT / PEFT / preference / reasoning
├── inference/             # 推理：serving / api / evaluation / loaders / apps
├── tools/                 # 独立工具库：distributed
├── models/                # 共享模型定义
├── layers/                # 共享层、tokenizer、生成原语
├── config/、utils/        # 共享配置与工具
├── dataset/               # 本地数据（gitignored）
├── tests/                 # unit / integration
├── docs/                  # 文档中心
├── scripts/               # 仓库级工具脚本
├── run_gpt2_124M.py       # GPT-2 预训练入口
├── run_llama.py           # Llama2 预训练入口（tiny 演示配置）
├── run_qwen.py            # Qwen3 预训练入口（tiny 演示配置）
└── run_bert.py            # BERT 预训练入口（MLM + NSP）
```

## 2. 三个主域与工具库

### 2.1 `pre_training/`

| 目录 | 职责 |
|---|---|
| `data_provider/` | 语料加载、Gutenberg 数据准备（`pgcorpus_gutenberg/`）、BERT 数据层（`bert/`） |
| `experiments/` | `Model_Pretrain`、超参搜索、bert 预训练实验 |
| `examples/` | 书本 walkthrough、GPT-2 小训练、FLOPs 分析、Gutenberg 独立训练示例 |
| `references/` | nanotron 配置参考 |
| `scripts/` | 单卡 / DP / DDP 启动 |

入口约定：**预训练主入口放根目录**（`run_gpt2_124M.py`、`run_bert.py`、`run_llama.py`、`run_qwen.py`，
均复用 `run_gpt2_124M.py` 的 parser 与 `pre_training.experiments.exp_pretrain_gpt2_124M` 主流程）；
后训练与推理的入口放域内（如 `post_training/sft/run_sft.py`）。

### 2.2 `post_training/`

| 目录 | 职责 |
|---|---|
| `sft/` | 分类微调、指令 SFT、SFT 评测；`data_provider`、`exp`、`layers`、`run`、`scripts` |
| `peft/` | LoRA/QLoRA/adapter/prefix/prompt、蒸馏实验；`model_peft`、`layers`、`data_provider/adapter`、`cfgs` |
| `preference/` | DPO/GRPO/PPO；`data_provider/dpo`、`exp`、`model_rlhf` |
| `reasoning/` | `training/`、`inference/`、`evaluation/` |
| `common/` | 共享 `data_loader`、`instruction_format`、`exp_basic` |

入口：

```bash
python post_training/sft/run_sft.py --help
python post_training/sft/run/run_gpt_instruction_sft.py
python post_training/sft/run/run_gpt_clf_sft.py
python post_training/peft/run_peft.py
python post_training/preference/model_rlhf/dpo/run_gpt_dpo.py
```

### 2.3 `inference/`

| 目录 | 职责 |
|---|---|
| `serving/` | `nano_vllm` serving 级实现 |
| `api/` | Ollama / OpenAI 客户端 |
| `evaluation/` | 指令回答评测 |
| `loaders/` | GPT-2 / Qwen3 / Gemma3 / Llama 权重加载 |
| `apps/` | Chainlit UI |
| `examples/` | GPT-2 生成示例 |
| `scripts/` | UI / 维护脚本 |

入口：

```bash
bash inference/scripts/run_app_own.sh
python inference/loaders/qwen3/qwen3_06B_kvcache_load.py
```

## 2.4 `tools/`

| 目录 | 职责 |
|---|---|
| `distributed/` | DP/DDP/FSDP/TP/DeepSpeed/fairscale 参考实现 |
| `profiling/` | 参数量、吞吐、显存、FLOPs/MFU |
| `checkpoint/` | 原子保存/恢复、轮转 |
| `data/` | JSONL、去重、packing |
| 未来 | 量化、benchmark、tracking、部署等 |

工具库不依赖三个主域的内部实现，详见 [70-tools](../70-tools/index.md)。

## 3. 共享资产

| 目录 | 内容 |
|---|---|
| `models/` | gpt2、llama、qwen3、gemma3、bert、deepseek |
| `layers/` | attention、block、FFN、norm、RoPE、MoE、KV cache、generation、tokenizers |
| `config/` | canonical 模型超参表（gpt2/llama/qwen3），供权重加载与示例使用；训练入口走 argparse 不经 config/。config 模块禁止 import 副作用（device/种子/tokenizer 在调用方获取）；权重下载在 `inference/loaders/qwen3/model_download.py` |
| `utils/` | optimizer、LR、loss、logging、device、DDP、checkpoint |

`layers/inference.py`、`layers/kv_cache.py` 留在共享层：既是教材实现，也是预训练采样所需；`inference/serving/nano_vllm` 负责 serving 优化。

## 4. 测试

- `tests/unit/`：单模块、无网络、秒级。
- `tests/integration/`：跨模块/端到端（当前为占位）。
- 运行：`python -m unittest discover -s tests -v`

## 5. 迁移说明

来自 `TinyLLMFinetuning` 的内容已拆为 `post_training/{sft,peft,preference}`；来自 `TinyLLMInference` 的内容拆为 `inference/{serving,api,evaluation,loaders,apps}`。历史记录见 [migration-2026-10.md](../60-reference/migration-2026-10.md)。

## 6. 已知遗留

- 部分旧脚本仍依赖缺失的 `utils.llm`，属于迁移前遗留。
- 2026-10-18（二）：根级 `data_provider/` 已删除，`bert/` 并入 `pre_training/data_provider/bert/`；
  各域数据目录统一命名为 `data_provider/`。通用数据算法在 `tools/data/`。
- 2026-10-18：`pre_training/legacy/` 与 `analysis/` 已解散；Llama/Qwen 入口修复并提升到根目录；
  FLOPs 分析并入 `pre_training/examples/`。
