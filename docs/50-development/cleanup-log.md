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
- 2026-10-20 推理域清理（P1）：
    - 删除死文件：`inference/loaders/qwen3/load_weights_moe.py`（与 load_weights.py 逐行重复且无引用）、
      `inference/scripts/update_codes.sh`（旧仓子模块时代遗物）、`inference/scripts/rm_pyc.sh`（sudo 删除危险且冗余）；
    - 修复 10 处 bug：app_orig 缺 import/None 下标、app_own decode 参数错、openai_evaluate stale import
      （`utils.llm.` → `utils.`）、llama2/llama3 eos id 错误（50256 → 2 / 128001）、llama3 instruct 用错 repo_id、
      llama2/3 下载缓存路径不一致、gemma3 权重误下载到 tokenizer 目录、qwen3 model_download 死分支、
      gpt2_inference 的 `eos_id` kwarg 错误；
    - `openai_gpt2_weights_load.py` 的 tensorflow 改为懒加载（未安装不影响其他加载路径）；
    - `run_app_*.sh` 修复相对路径（脚本内 cd 到 inference/）；
    - 修复断链调用方 `post_training/sft/exp/exp_finetune_gpt_instruction_flow_evaluate.py` 的双 stale import。
- 2026-10-20 推理域合并（P2）：
    - LLM judge 合并：新增 `inference/evaluation/judge.py`（`backend="ollama"|"openai"` 统一入口），
      `ollama_evaluate.py` / `openai_evaluate.py` 改为兼容 shim（原签名不变，内部委托 judge.py）；
    - Chainlit app 合并：新增 `inference/apps/app.py`（`GPT2_WEIGHTS=orig|own` 选择权重来源），
      `app_orig.py` / `app_own.py` 改为兼容 shim（设置环境变量后 import app 注册 handler）；
    - 启动脚本合并：`run_app_orig.sh` / `run_app_own.sh` 合并为 `run_app.sh`（接受 orig|own 参数），旧脚本已删除。
- 2026-10-20 推理域通用化（P3）：
    - 新增公共 API：`inference/registry.py`（MODEL_REGISTRY）、`inference/loading.py`（`load_model` 统一加载）、
      `inference/generation.py`（`generate_text`/`generate_text_stream` 统一生成 facade）、
      `inference/__init__.py`（PEP 562 懒加载导出）；
    - nano_vllm 包化：库模块全部改相对 import、删除 sys.path 操作（不再遮蔽仓根 `config/` 等同名包），
      新增包级 `__init__.py`（导出 LLM/Config/SamplingParams）与 `llm.py` 便捷入口，
      入口脚本保留直跑/-m 双模式；
    - 新增 `tests/unit/test_inference_registry.py`（注册表、load_model 错误路径、facade 离线生成、懒加载契约）。

## 2. 数据

- `dataset/` 从约 76 GB 清理到约 5.8 GB。
- 删除重复压缩包、HF cache、MMLU 辅助训练、IMDB 原始目录、DISC-Law QA 子集、CLUENER、通用 SFT 大语料、mobvoi 原始 JSONL 等。
- 保留并归类为 `pretraining/`、`finetuning/`、`evaluation/`；布局与代码路径见 [datasets](datasets.md)。

## 3. 文档

- 文档统一到 `docs/`，按 00–70 编号组织。
- 删除一次性数据集审计页、MMLU 上游 README 副本、旧“现状盘点/路径索引”页。
- `LLMs-from-scratch` 覆盖度分析保留为历史快照，当前架构以 [architecture](../00-overview/architecture.md) 为准。

## 4. 仍存在的遗留

- `post_training/sft/run_sft.py` 的 `TrlParser` 用法与当前 `trl` 版本不兼容。
- `post_training/peft/.../lora_dora_from_scratch.py` 引用不存在的 `data_provider.lora`。
- `post_training/peft/run_peft.py` 会尝试下载 HF 模型，离线/受限环境不可直接运行。

## 5. 验证

```bash
python -m unittest discover -s tests -v   # 67 项（2026-10-20）
python run_gpt2_124M.py --help            # OK
python scripts/check_docs.py              # OK
```
