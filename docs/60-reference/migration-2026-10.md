> 所属：[TinyLLM Docs](../README.md) › [Reference](index.md)

# 2026-10 架构迁移记录

> 本次迁移把 `TinyLLMFinetuning`、`TinyLLMInference` 两个仓库并入 TinyLLM，并把仓库重构为  
> **pre_training / post_training / inference** 三个域。

## 1. 来源

| 仓库 | 路径 | commit | 文件数 | 大小 |
|---|---|---|---:|---:|
| TinyLLMFinetuning | `~/projects/llm_proj/TinyLLMFinetuning` | `07dcbde` | 144 | 19 MB |
| TinyLLMInference | `~/projects/llm_proj/TinyLLMInference` | `b01be3b` | 20 | 360 KB |

两个仓库在迁移前 `git status` 均为 clean。

## 2. 目标映射

### 2.1 预训练

| 原位置 | 新位置 |
|---|---|
| `data_provider/` | `pre_training/data_provider/` |
| `exp/` | `pre_training/exp/` |
| `scripts/pretrain_gpt2_124M*.sh` | `pre_training/scripts/` |
| `test/`（书本示例） | `pre_training/examples/`、`pre_training/analysis/` |
| `todo/pretraining/` | `pre_training/extras/` |

### 2.2 后训练

| 原位置 | 新位置 |
|---|---|
| `TinyLLMFinetuning/*` | `post_training/finetuning/` |
| `qwen3_reasoning/*` | `post_training/reasoning/` |
| `todo/finetuning/llmbox` | `post_training/finetuning/data_provider/llmbox/` |
| `todo/finetuning/minimind` | ~~`post_training/imported/minimind/`~~（已删除，见 `reference-projects.md`） |
| `todo/finetuning/loaders/` | `inference/loaders/gpt2/` |
| `todo/finetuning/scripts/` | `post_training/finetuning/scripts/` |
| `todo/finetuning/datasets/THUCNews` | `dataset/finetuning/THUCNews/` |
| `models/bert`（原 `todo/finetuning/bert`） | `models/bert/` |

### 2.3 推理

| 原位置 | 新位置 |
|---|---|
| `TinyLLMInference/*` | `inference/toolkit/` |
| `todo/inference/apps/` | `inference/apps/` |
| `todo/inference/loaders/qwen3_model/` | `inference/loaders/qwen3/` |
| `todo/inference/loaders/gemma3_model/` | `inference/loaders/gemma3/` |
| `todo/inference/loaders/llama_load/` | `inference/loaders/llama/` |
| `todo/inference/tests/gpt_inference_test.py` | `inference/examples/gpt2_inference.py` |
| `todo/inference/scripts/` | `inference/scripts/` |

### 2.4 测试与 BERT

| 原位置 | 新位置 |
|---|---|
| `test/attention_test.py` 等 | `pre_training/examples/` |
| `test/flops_analysis.py` | `pre_training/analysis/` |
| `test/gpt2_model/gpt2_load/` | `inference/loaders/gpt2/` |
| `test/gpt2_model/local_test/` | `pre_training/examples/gpt2_book/` |
| `test/`（目录本身） | 删除，改为 `tests/` 单元测试 |
| `todo/finetuning/bert/` | `models/bert/`（`todo/` 子目录改名为 `experiments/`） |

## 3. 导入路径修复

- 根 `run_gpt2_124M.py`：`exp.exp_pretrain_gpt2_124M` → `pre_training.exp.exp_pretrain_gpt2_124M`。
- `pre_training/exp/exp_pretrain_gpt2_124M.py`：
  - `data_provider.data_loader` → `pre_training.data_provider.data_loader`；
  - `exp.exp_basic` → `pre_training.exp.exp_basic`。
- `pre_training/examples/gpt2_pretrain_demo.py`、`text_embedding.py`：同步更新。
- `pre_training/scripts/pretrain_gpt2_124M_DDP.sh`、`_DP.sh`：更新 `exp/` 路径。
- `pre_training/extras/legacy_entrypoints/run_llama.py`、`run_qwen.py`：`exp...` → `pre_training.exp...`。
- `post_training/reasoning/evaluation/...`：`qwen3_reasoning.llm_basic...` → `post_training.reasoning.llm_basic...`。
- `inference/loaders/qwen3/*`：`test.qwen3_model.load_weights` → `inference.loaders.qwen3.load_weights`。
- `inference/loaders/llama/*`：旧的 `models.gpt2_model_cfg` / `models.llama_load` / `models.llama2` 路径改为 `config.llama` / `inference.loaders.llama` / `models.llama.*`。

## 4. 验证

| 检查 | 结果 |
|---|---|
| `python -m compileall -q pre_training post_training inference tests models layers utils` | 通过（仅遗留 SyntaxWarning） |
| 核心 import（pre_training / reasoning / inference loaders） | 通过 |
| `python run_gpt2_124M.py --help` | 通过 |
| `python -m unittest discover -s tests -v` | 14/14 通过 |
| 源仓库文件清单 vs 目标文件清单 | TinyLLMFinetuning 144/144、TinyLLMInference 20/20（非 ASCII 文件名已核对） |

## 5. 兄弟仓库清理

- 两个仓库的内容已复制到本仓库；`.git` 保留在各自仓库，历史仍可用 `git log` / `git checkout` 恢复。
- 两个仓库的工作区已清空，仅保留 `.git`。
- 如需完全删除，请确认后再执行（本次不删除 `.git`）。

## 6. 已知遗留

1. `post_training/finetuning/` 保留原 `TinyLLMFinetuning` 的绝对导入，部分脚本需要 `PYTHONPATH` 才能运行。
3. `layers/tokenizers/` 已 vendor 进主仓库；原嵌套仓库历史保存在 `docs/archive/tokenizers-premerge-2026-10-18.bundle`。
5. `pre_training/extras/legacy_experiments/` 与 `legacy_entrypoints/` 仍为空壳/失修状态。
