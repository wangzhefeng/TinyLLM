# 清理记录

> 所属：[TinyLLM Docs](../README.md) › [Development](index.md)
> 更新时间：2026-10-18。本文只记录**当前仍然成立**的清理结果。

## 1. 代码与目录

- 三个主域重组为 `pre_training/`、`post_training/`、`inference/`。
- `post_training/` 拆为 `sft/`、`peft/`、`preference/`、`reasoning/`、`common/`。
- `inference/` 拆为 `serving/`、`api/`、`evaluation/`、`loaders/`、`apps/`、`examples/`。
- 分布式训练独立为 `tools/distributed/`；新增 `tools/profiling/`、`tools/checkpoint/`、`tools/data/`。
- 删除空壳、重复实现和未接入的外部参考，详见 [reference-projects](reference-projects.md)。

## 2. 数据

- `dataset/` 从约 76 GB 清理到约 5.8 GB。
- 删除重复压缩包、HF cache、MMLU 辅助训练、IMDB 原始目录、DISC-Law QA 子集、CLUENER、通用 SFT 大语料、mobvoi 原始 JSONL 等。
- 保留并归类为 `pretraining/`、`finetuning/`、`evaluation/`；布局与代码路径见 [datasets](datasets.md)。

## 3. 文档

- 文档统一到 `docs/`，按 00–70 编号组织。
- 删除一次性数据集审计页、MMLU 上游 README 副本、旧“现状盘点/路径索引”页。
- `LLMs-from-scratch` 覆盖度分析保留为历史快照，当前架构以 [architecture](../00-overview/architecture.md) 为准。

## 4. 仍存在的遗留

- `pre_training/legacy/` 的 `run_llama.py`、`run_qwen.py` 与 `Model_Pretrain` 签名不匹配。
- 部分旧脚本仍依赖缺失的 `utils.llm`、`models.gpt`。
- `post_training/sft/run_sft.py` 的 `TrlParser` 用法与当前 `trl` 版本不兼容。
- `post_training/peft/run_peft.py` 会尝试下载 HF 模型，离线/受限环境不可直接运行。

## 5. 验证

```bash
python -m unittest discover -s tests -v   # 24/24
python run_gpt2_124M.py --help            # OK
python scripts/check_docs.py              # OK
```
