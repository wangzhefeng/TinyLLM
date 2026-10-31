# 代码 → 文档映射

> 所属：[TinyLLM Docs](../README.md) › [Overview](index.md)

修改代码时，按下表找到应同步更新的文档。表按顶层目录逐区对照（2026-10-25 核对）。

| 代码区域 | 文档 |
|---|---|
| `pre_training/data_provider/`（含 `bert/`） | [data](../10-pretraining/data.md)、[datasets](../50-development/datasets.md) |
| `pre_training/experiments/`（`exp_basic.py`、`exp_component_ablation.py`、`bert/dapt_data.py`）、根 `run_*.py` | [training](../10-pretraining/training.md)、[ablation](../10-pretraining/ablation.md) |
| `pre_training/examples/{components,training,analysis}/` | [examples](../10-pretraining/examples.md) |
| `pre_training/references/{gpt2_book,pgcorpus,nanotron}/` | [extras](../10-pretraining/extras.md)、[gutenberg](../10-pretraining/gutenberg.md) |
| BERT 数据/模型/实验与示例 | [bert](../40-models/bert.md) |
| `training/`（trainer/checkpoint/callbacks） | [architecture](architecture.md)、[training](../10-pretraining/training.md)、[checkpoint](../70-tools/checkpoint.md) |
| `post_training/sft/`（含 `evaluation/judge.py`） | [finetuning](../20-post-training/finetuning/index.md) |
| `post_training/peft/`（含 `demos/`） | [peft](../20-post-training/peft.md) |
| `post_training/preference/`（dpo/rm/run） | [dpo](../20-post-training/dpo.md)、[rlhf](../20-post-training/rlhf.md) |
| `post_training/rl/`（grpo/ppo/reinforce） | [rlhf](../20-post-training/rlhf.md)、[reasoning](../20-post-training/reasoning.md) |
| `post_training/reasoning/`（训练/蒸馏） | [reasoning](../20-post-training/reasoning.md)、[distillation](../20-post-training/distillation.md) |
| `post_training/references/`（trl_sft/trl_grpo/hf_peft/unsloth） | [peft](../20-post-training/peft.md)、[cleanup-log](../50-development/cleanup-log.md) |
| `evaluation/clients/`、`verifiers/`、`judges/` | [评测公共库](../35-evaluation/index.md) |
| `inference/loading.py`、`generation.py`、`generate_core.py`、`models/registry.py` | [inference](../30-inference/index.md) |
| `inference/interpretability.py` | [interpretability](../30-inference/interpretability.md) |
| `inference/harness/`（math500/mmlu） | [evaluation](../30-inference/evaluation.md)、[评测公共库](../35-evaluation/index.md) |
| `inference/strategies/` | [reasoning](../20-post-training/reasoning.md)、[scaling](../30-inference/reasoning-scaling.md) |
| `inference/serving/`、`inference/apps/`、`inference/scripts/` | [serving](../30-inference/serving.md)、[nano-vllm](../30-inference/nano-vllm.md) |
| `models/io/`（权重 IO 与 HF 导出） | [loaders](../30-inference/loaders.md) |
| `models/`（各模型族 + `registry.py`） | [models](../40-models/index.md) 下对应页面 |
| `layers/`（含 `tokenizers/`、`quantization/nf4.py`） | [architecture](../40-models/architecture.md)、[tokenizers](../40-models/tokenizers.md)、[peft](../20-post-training/peft.md) |
| `config/` | [architecture](architecture.md)、对应模型页 |
| `utils/`（含 `online.py`） | [utils](../50-development/utils.md) |
| `tools/data/`（含 `read_text_label_tsv`）、`tools/checkpoint/` | [data](../70-tools/data.md)、[checkpoint](../70-tools/checkpoint.md) |
| `tools/distributed/` | [distributed](../70-tools/distributed.md)、[frameworks](../70-tools/distributed-frameworks.md)、[roadmap](../70-tools/distributed-roadmap.md) |
| `tools/profiling/` | [profiling](../70-tools/profiling.md) |
| `tests/`、`.github/workflows/` | [testing](../50-development/testing.md) |
| `dataset/` | [datasets](../50-development/datasets.md) |
| `results/` | [architecture](architecture.md)、[outputs](../50-development/outputs.md) |
| `logs/`、`results/` 产物约定 | [outputs](../50-development/outputs.md) |
| 文档结构 | [docs-maintenance](../50-development/docs-maintenance.md)、[AGENTS](../../AGENTS.md) |

## 规则

- 代码移动/删除时，先更新本表、相关页面和 [cleanup-log](../50-development/cleanup-log.md)。
- 新增域或顶层目录时，更新根 README 与 docs/README。
- 当前不再保留兼容 shim：旧 `inference/{loaders,api,evaluation,registry}`、`post_training/common`、
  `model_rlhf` / `model_peft` 等路径均已消亡，新代码一律使用上表新路径。
