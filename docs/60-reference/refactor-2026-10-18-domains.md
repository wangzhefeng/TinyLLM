# 三域内部结构重构（2026-10-18）

> 所属：[TinyLLM Docs](../README.md) › [Reference](index.md)

> **历史快照说明（2026-10-25 追加）**：本文记录 2026-10-18 重构；此后结构又有两轮演进
> （2026-10-24 共享运行时、2026-10-25 评测公共库上提）。文中出现的 `inference/api`、
> `inference/evaluation`、`model_rlhf` 等路径属历史路径，现已迁移到 `evaluation/` 公共库、
> `post_training/references/` 等，当前结构见 [architecture](../00-overview/architecture.md)。

## 1. 重构前的问题

| 域 | 问题 |
|---|---|
| `pre_training/` | `extras/` 同时放分布式、nanotron 参考、失修入口；`exp/` 名称过短，职责不直观 |
| `post_training/` | `finetuning/` 是一个大杂烩：SFT、分类、PEFT、DPO/RLHF、脚本、数据全混在一起 |
| `inference/` | `toolkit/` 同时放 serving（nano_vllm）、API 客户端、评测 |

## 2. 重构后的结构

### pre_training

```text
pre_training/
├── data/            # 数据加载、Gutenberg
├── experiments/     # 实验类、超参
├── examples/        # 书本示例
├── analysis/        # FLOPs
├── references/      # nanotron
├── legacy/          # 失修入口
└── scripts/
```

### post_training

```text
post_training/
├── sft/          # 分类 + 指令 SFT + 评测
├── peft/         # LoRA/QLoRA/adapter/蒸馏实验
├── preference/   # DPO/GRPO/PPO
├── reasoning/    # training / inference / evaluation
└── common/       # 共享 data_loader/instruction_format/exp_basic
```

### inference

```text
inference/
├── serving/      # nano_vllm
├── api/          # Ollama/OpenAI 客户端
├── evaluation/   # 指令回答评测
├── loaders/      # 权重加载
├── apps/         # Chainlit
├── examples/
└── scripts/
```

## 3. 入口变化

| 旧 | 新 |
|---|---|
| `pre_training/exp/...` | `pre_training/experiments/...` |
| `pre_training/data_provider/...` | `pre_training/data/...` |
| `post_training/finetuning/run_sft.py` | `post_training/sft/run_sft.py` |
| `post_training/finetuning/run_peft.py` | `post_training/peft/run_peft.py` |
| `post_training/finetuning/model_rlhf/dpo/run_gpt_dpo.py` | `post_training/preference/model_rlhf/dpo/run_gpt_dpo.py` |
| `inference/toolkit/nano_vllm` | `inference/serving/nano_vllm` |
| `inference/toolkit/inference_utils` | `inference/api` |
| `inference/toolkit/evaluate` | `inference/evaluation` |

## 4. 后续

2026-10-18 后续：`distributed/` 已独立为顶层工具库 `tools/distributed/`，不属于 `pre_training/`。

## 5. 说明

- 旧代码中 `utils.llm`、`models.gpt` 等缺失依赖是迁移前遗留，本次未修复。
- `pre_training/legacy/` 的 Llama/Qwen 入口仍与 `Model_Pretrain` 签名不匹配。
- 本次只重构结构，不改变主链路行为；测试与 `run_gpt2_124M.py --help` 已验证。
