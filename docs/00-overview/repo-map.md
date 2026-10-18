# 代码 → 文档映射

> 所属：[TinyLLM Docs](../README.md) › [Overview](index.md)

修改代码时，按下表找到应同步更新的文档。

| 代码区域 | 文档 |
|---|---|
| `pre_training/data/` | [10-pretraining/data.md](../10-pretraining/data.md)、[datasets](../50-development/datasets.md) |
| `pre_training/experiments/`、`run_gpt2_124M.py` | [10-pretraining/training.md](../10-pretraining/training.md) |
| `pre_training/examples/` | [10-pretraining/examples.md](../10-pretraining/examples.md) |
| `pre_training/references/`、`pre_training/legacy/` | [10-pretraining/extras.md](../10-pretraining/extras.md) |
| `tools/distributed/` | [70-tools/distributed.md](../70-tools/distributed.md) |
| `tools/profiling/` | [70-tools/profiling.md](../70-tools/profiling.md) |
| `tools/checkpoint/` | [70-tools/checkpoint.md](../70-tools/checkpoint.md) |
| `tools/data/` | [70-tools/data.md](../70-tools/data.md) |
| `post_training/sft/` | [20-post-training/finetuning/index.md](../20-post-training/finetuning/index.md) |
| `post_training/peft/` | [20-post-training/peft.md](../20-post-training/peft.md) |
| `post_training/preference/` | [20-post-training/dpo.md](../20-post-training/dpo.md)、[rlhf.md](../20-post-training/rlhf.md) |
| `post_training/common/` | [20-post-training/finetuning/index.md](../20-post-training/finetuning/index.md) |
| `post_training/reasoning/` | [20-post-training/reasoning.md](../20-post-training/reasoning.md) |
| `inference/serving/`、`inference/api/` | [30-inference/serving.md](../30-inference/serving.md) |
| `inference/evaluation/` | [30-inference/evaluation.md](../30-inference/evaluation.md) |
| `inference/loaders/` | [30-inference/loaders.md](../30-inference/loaders.md) |
| `inference/apps/` | [30-inference/index.md](../30-inference/index.md) |
| `models/` | [40-models/index.md](../40-models/index.md) 下对应页面 |
| `layers/` | [40-models/architecture.md](../40-models/architecture.md)、[tokenizers.md](../40-models/tokenizers.md) |
| `utils/` | [50-development/utils.md](../50-development/utils.md) |
| `tests/` | [50-development/testing.md](../50-development/testing.md) |
| `dataset/` | [50-development/datasets.md](../50-development/datasets.md) |
| 文档结构本身 | [50-development/docs-maintenance.md](../50-development/docs-maintenance.md)、[AGENTS.md](../../AGENTS.md) |

## 规则

- 代码移动/删除时，先更新本表和相关页面链接。
- 新增域或顶层目录时，更新 [根 README](../../README.md) 与 [docs/README.md](../README.md)。
- 不确定该写在哪时，先放在对应 `index.md` 的“待整理”区，再拆页。
