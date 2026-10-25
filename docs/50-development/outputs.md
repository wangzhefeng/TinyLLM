> 所属：[TinyLLM Docs](../README.md) › [Development](index.md)

# 输出产物路径约定

一句话判据：**结论性产物放 `results/`，过程性输出放 `logs/`**。
`logs/`、`results/`、`downloaded_models/` 三者均已被 `.gitignore` 覆盖，不入库。

## 目录表

| 目录 | 放什么 | 谁写入 |
|---|---|---|
| `logs/` | 过程性输出：log_util 服务日志（`logs/main`、`logs/<experiment>/service*`）、训练遥测（metrics CSV/PNG） | `utils/log_util`；各训练入口的 `--logs` 参数 |
| `results/checkpoints/<task>/` | 训练权重 checkpoint（会被再次加载或复用） | 各训练入口的 `--out` 参数（grpo、distill、examples 等） |
| `results/math500/` | MATH-500 评测报告 JSON | `inference/harness/math500.py`（`--out`） |
| `results/exports/` | HF 格式导出（预留，暂不建目录） | `models/io/export/` |
| `downloaded_models/` | 外部权重缓存（下载产物，不手改、不迁移） | 各 loader 的下载脚本 |

## 主要入口默认值

| 入口 | 参数 | 默认值 |
|---|---|---|
| `post_training/rl/run.py` | `--out` | `results/checkpoints/grpo` |
| `post_training/rl/run.py` | `--logs` | `logs/grpo` |
| `post_training/reasoning/run.py` | `--out` | `results/checkpoints/distill` |
| `post_training/reasoning/run.py` | `--logs` | `logs/distill` |
| `post_training/sft/run/run_gpt_{clf,instruction}_sft.py` | `--finetuned_model_path` / `--checkpoints` | `results/checkpoints/sft` |
| `post_training/preference/run/run_gpt_dpo.py` | `--checkpoints` / `--model_path` | `results/checkpoints/dpo`（输入 `--finetuned_model_path` 指向 `results/checkpoints/sft/`） |
| `inference/harness/math500.py` | `--out` | `results/math500/<model>_<strategy>.json` |
| `inference/apps/app.py` | `GPT2_CHECKPOINT` | `results/checkpoints/gpt2/model.pth` |

默认值均可用对应参数或环境变量覆盖；报告与 checkpoint 的写入函数会自动创建父目录。

## 外部参考示例例外

`post_training/references/hf_peft/run_peft.py` 与 `references/unsloth/` 中的字符串字面量目录
（如 `save_pretrained("output_dir")`、`"lora_model"`）是外部参考/教程示例，
不遵循本约定，也不做迁移。

## 相关

- [architecture](../00-overview/architecture.md)
- [datasets](datasets.md)
- [evaluation](../30-inference/evaluation.md)
- [cleanup-log](cleanup-log.md)
