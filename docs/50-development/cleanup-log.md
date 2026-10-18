# 清理记录

> 所属：[TinyLLM Docs](../README.md) › [Development](index.md)
> 更新时间：2026-10-18。本文只记录**当前仍然成立**的清理结果。

## 1. 代码与目录

- 三个主域重组为 `pre_training/`、`post_training/`、`inference/`。
- `post_training/` 拆为 `sft/`、`peft/`、`preference/`、`reasoning/`、`common/`。
- `inference/` 拆为 `serving/`、`api/`、`evaluation/`、`loaders/`、`apps/`、`examples/`。
- 分布式训练独立为 `tools/distributed/`；新增 `tools/profiling/`、`tools/checkpoint/`、`tools/data/`。
- 删除空壳、重复实现和未接入的外部参考，详见 [reference-projects](reference-projects.md)。
- 2026-10-18 预训练域归位：
    - 解散 `pre_training/legacy/`：`run_llama.py` / `run_qwen.py` 修复后提升为根目录入口；
    - 解散 `pre_training/analysis/`：`flops_analysis.py` 并入 `pre_training/examples/`（可复用 FLOPs/MFU 实现在 `tools/profiling/`）；
    - 拆分 `data/pgcorpus_gutenberg/`：`pretraining_simple.py` → `examples/pgcorpus_pretraining_simple.py`，`tests.py` → `tests/integration/test_pgcorpus_gutenberg.py`（改为 unittest，缺上游包时跳过）；
    - 删除无引用且无法导入的 `models/gpt2/gpt.py`；
    - 修复 `layers/tokenizers/` 三个 demo 对已删除 `data_provider.data_loader` 的引用（改用 `utils.load_save_data.load_local_data`）。
- 2026-10-18 数据层归位：删除根级 `data_provider/`（假约定的"共享数据层"，实际只剩 bert），
  `bert/` 并入 `pre_training/data_provider/bert/`；`pre_training/data/` 更名为 `pre_training/data_provider/`，
  与 post_training 各子域的 `data_provider/` 命名统一；通用数据算法集中在 `tools/data/`。
- 2026-10-18 config/ 清理：
    - `config/gpt2/model_cfgs.py` 去除 import 副作用（`device_setting()`、`torch.manual_seed(123)`、
      tiktoken 下载），device/tokenizer 移到 5 个调用点本地获取；
    - `config/qwen3/model_download.py` 移到 `inference/loaders/qwen3/model_download.py`（下载是 loader 职责），
      并修复 `"instruction"` 分支重复 `"model"` 键的 bug；
    - 全仓修复 `max_new_toknes` 拼写（含 `inference/examples/gpt2_inference.py` 一处运行即 TypeError 的调用）。

## 2. 数据

- `dataset/` 从约 76 GB 清理到约 5.8 GB。
- 删除重复压缩包、HF cache、MMLU 辅助训练、IMDB 原始目录、DISC-Law QA 子集、CLUENER、通用 SFT 大语料、mobvoi 原始 JSONL 等。
- 保留并归类为 `pretraining/`、`finetuning/`、`evaluation/`；布局与代码路径见 [datasets](datasets.md)。

## 3. 文档

- 文档统一到 `docs/`，按 00–70 编号组织。
- 删除一次性数据集审计页、MMLU 上游 README 副本、旧“现状盘点/路径索引”页。
- `LLMs-from-scratch` 覆盖度分析保留为历史快照，当前架构以 [architecture](../00-overview/architecture.md) 为准。

## 4. 仍存在的遗留

- 部分旧脚本仍依赖缺失的 `utils.llm`。
- `post_training/sft/run_sft.py` 的 `TrlParser` 用法与当前 `trl` 版本不兼容。
- `post_training/peft/.../lora_dora_from_scratch.py` 引用不存在的 `data_provider.lora`。
- `post_training/peft/run_peft.py` 会尝试下载 HF 模型，离线/受限环境不可直接运行。

## 5. 验证

```bash
python -m unittest discover -s tests -v   # 24/24
python run_gpt2_124M.py --help            # OK
python scripts/check_docs.py              # OK
```
