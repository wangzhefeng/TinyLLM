# 代码 → 文档映射

> 所属：[TinyLLM Docs](../README.md) › [Overview](index.md)

修改代码时，按下表找到应同步更新的文档。

| 代码区域 | 文档 |
|---|---|
| `pre_training/data_provider/` | [data](../10-pretraining/data.md)、[datasets](../50-development/datasets.md) |
| `pre_training/experiments/`、根 `run_*.py` | [training](../10-pretraining/training.md) |
| BERT 数据/模型/实验与示例 | [bert](../40-models/bert.md) |
| `pre_training/examples/` | [examples](../10-pretraining/examples.md) |
| `pre_training/references/` | [extras](../10-pretraining/extras.md) |
| `training/` | [architecture](architecture.md)、[training](../10-pretraining/training.md)、[checkpoint](../70-tools/checkpoint.md) |
| `post_training/sft/` | [finetuning](../20-post-training/finetuning/index.md) |
| `post_training/peft/` | [peft](../20-post-training/peft.md) |
| `post_training/preference/` | [dpo](../20-post-training/dpo.md)、[rlhf](../20-post-training/rlhf.md) |
| `post_training/rl/` | [rlhf](../20-post-training/rlhf.md)、[reasoning](../20-post-training/reasoning.md) |
| `post_training/reasoning/training/` | [reasoning](../20-post-training/reasoning.md) |
| `post_training/reasoning/data_gen.py`、`run.py` | [distillation](../20-post-training/distillation.md) |
| `inference/registry.py`、`loading.py`、`generation.py`、`generate_core.py` | [inference](../30-inference/index.md) |
| `inference/strategies/` | [reasoning](../20-post-training/reasoning.md)、[inference](../30-inference/index.md) |
| `inference/serving/`、`inference/api/` | [serving](../30-inference/serving.md) |
| `inference/evaluation/` | [evaluation](../30-inference/evaluation.md)、[datasets](../50-development/datasets.md) |
| `inference/evaluation/harness/` | [evaluation](../30-inference/evaluation.md) |
| `models/io/`、`models/io/export/`、`inference/loaders/`（兼容 shim） | [loaders](../30-inference/loaders.md) |
| `models/` | [models](../40-models/index.md) 下对应页面 |
| `layers/` | [architecture](../40-models/architecture.md)、[tokenizers](../40-models/tokenizers.md) |
| `config/` | [architecture](architecture.md)、对应模型页 |
| `utils/` | [utils](../50-development/utils.md) |
| `tools/distributed/` | [distributed](../70-tools/distributed.md)、[frameworks](../70-tools/distributed-frameworks.md)、[roadmap](../70-tools/distributed-roadmap.md) |
| `tools/profiling/` | [profiling](../70-tools/profiling.md) |
| `tools/checkpoint/` | [checkpoint](../70-tools/checkpoint.md) |
| `tools/data/` | [data](../70-tools/data.md) |
| `tests/`、`.github/workflows/` | [testing](../50-development/testing.md) |
| `dataset/` | [datasets](../50-development/datasets.md) |
| `results/` | [architecture](architecture.md) |
| `logs/`、`results/` 产物约定 | [outputs](../50-development/outputs.md) |
| 文档结构 | [docs-maintenance](../50-development/docs-maintenance.md)、[AGENTS](../../AGENTS.md) |

## 规则

- 代码移动/删除时，先更新本表、相关页面和 cleanup-log。
- 新增域或顶层目录时，更新根 README 与 docs/README。
- 新代码使用 `models/io`、`post_training/rl`、`inference/strategies`；旧路径仅为兼容 shim。
