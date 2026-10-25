# 清理记录

> 所属：[TinyLLM Docs](../README.md) › [Development](index.md)
> 更新时间：2026-10-25。本文只记录**当前仍然成立**的清理结果；历史过程细节见
> [migration-2026-10](../60-reference/migration-2026-10.md) 与各域页面。

## 1. 代码与目录

- 三个主域重组为 `pre_training/`、`post_training/`、`inference/`；当前结构与职责以
  [architecture](../00-overview/architecture.md)、[repo-map](../00-overview/repo-map.md) 为准。
- 分布式训练独立为 `tools/distributed/`；新增 `tools/profiling/`、`tools/checkpoint/`、`tools/data/`。
- 删除空壳、重复实现和未接入的外部参考，详见 [reference-projects](reference-projects.md)。
- 2026-10-25 tokenizers 模块清理与协议强制化（本轮）：
    - Bug 修复：`bpe/llama3_8b_bpe.py` 默认路径锚定仓库根 `downloaded_models/`（原 CWD 相对），新增 `tokenizer_file_path=` 覆盖参数与 gated 仓库获取说明，`__all__` 由指向不存在类的 `["Tokenizer"]` 修正为 `["Llama38bTokenizer"]`；`bpe/openai_gpt2_bpe.py` 默认目录由不存在的 `layers/models/` 改锚 `downloaded_models/gpt2_model`；`bpe/simple_bpe.py` 落盘路径锚定 `layers/tokenizers/models/`（原 CWD 相对）；`sp/llama2_7b_sentencepiece.py` 的 `parents[4]` 越界修正（原解析到仓库外）与 `_download_tokenzier` 拼写修复（两个 bpe/sp 文件）。
    - 删除死代码与孤儿资产：`bpe/hf_gpt2_bpe.py`（`HuffingFaceGPT2BPE`，零外部引用且未注册）、`chat.py` 的 `format_input_phi3`、`apply_chat_template` 与 `format_gemma3_chat`（前者的调用方均在 post_training，后二者零引用）、`gemma.py` 的 `apply_chat_template` 与 `tokenizer_download` 透传别名、`images/tokenizer.png`；`layers/tokenizers/.gitignore` 由 220 行 boilerplate 收缩为仅 `models/`。
    - 对称化迁移（实现 = 子包）：`gemma.py` → `gemma3/gemma3_tokenizer.py`、`simple_custom.py` → `simple/simple_tokenizer.py`（类名与 registry key `"simple_custom"` 均不变，key 是 CLI 契约）。
    - 协议强制化：教学实现补齐 `TokenizerProtocol` 属性（`SimpleTokenizer`/`UnigramTokenizer`/`BPETokenizerSimple` 的 eos/pad 置 `None`、BPE 补 `n_vocab` property，`ExtendedTokenizer` eos/pad 委托被包装 tokenizer）；`create_tokenizer` 实例化后校验协议，缺成员抛 `TypeError` 并列出缺失项；`choose_tokenizer` 对无 kwargs 调用按归一化名称缓存实例。详见 [tokenizers](../40-models/tokenizers.md)。
- 2026-10-25 评测公共库上提与 references 分离（本轮；**所有兼容 shim 消亡**）：
    - 新增顶层 `evaluation/` 公共库：`clients/{ollama,openai,openrouter}.py`（合并重复 Ollama 客户端）、
      `verifiers/{math,parser}.py`（原 math_verifiers / advanced_parser）、
      `judges/{llm_judges,ollama,qwen3,leaderboards}.py`（leaderboard 三文件合并，见 [评测公共库](../35-evaluation/index.md)）；
    - 新增 `models/registry.py`（原 `inference/registry.py`；`pre_training/exp_basic.py` 硬编码注册表改为查它）、
      `layers/quantization/nf4.py`（原 `peft/layers/nf4.py`）、`utils/online.py`（HF Hub / wandb 登录）、
      `tools/data/io.py:read_text_label_tsv`（bert_sentiment 与 DAPT 共用）、`pre_training/__init__.py`；
    - `inference/harness/{math500,mmlu}.py`（原 `inference/evaluation/harness` 与 `benchmarks/multiple_choice`）；
      `post_training/sft/evaluation/judge.py`（原 `inference/evaluation/judge.py`，alpaca 指令打分）；
    - 新增 `post_training/references/` 外部栈对照区（保持上游原貌，不接入 `run.py` 分发器）：
      `trl_grpo/`（原 `preference/model_rlhf/grpo/` + bitsandbytes）、`trl_sft/`（原 `sft/run_sft.py` + peft 两组 QLoRA）、
      `unsloth/`（原 `peft/model_peft/unsloth_exps/`）、`hf_peft/`（原 `run_peft.py` + `model_peft/` 其余子目录）；
    - `post_training/rl/ppo.py`（原 `model_rlhf/ppo/main.py`）、`rl/reinforce.py`
      （原 `reasoning/training/reinforce.py`，解开 rl↔reasoning 环）、
      `preference/run/run_gpt_dpo.py`（原 `model_rlhf/dpo/`）、`peft/demos/`（lora_demo / prompt_tuning / scratch_exps）；
    - `pre_training/examples/` 三分组：`components/`、`training/`、`analysis/`；
      `pre_training/references/{gpt2_book,pgcorpus}/`（原 `examples/gpt2_book/` 与 `data_provider/pgcorpus_gutenberg/`）；
    - 删除：`inference/{loaders,api,evaluation}/` 整目录、`inference/registry.py`、
      `inference/strategies/llm_basic/`（qwen3_06B → `inference/examples/`）、
      `inference/apps/{app_orig,app_own}.py`、`inference/evaluation/{ollama_evaluate,openai_evaluate}.py` 垫片、
      `post_training/common/` 整目录（instruction_format/data_loader 垫片删、data_factory → trl_sft/sql_create_context 本地、
      exp_basic → demos/scratch_exps 本地、旧 shell 删）、`post_training/preference/model_rlhf/`、
      `post_training/peft/{model_peft,run}/`、`peft/layers/lora_dora.py`、
      `peft/layers/merge_adapte_weights.py`（改名 `merge_adapter_weights.py`）、
      `post_training/reasoning/{inference,evaluation}/` 别名目录与 `training/reinforcement_learning.py` 垫片、
      `pre_training/experiments/bert/{data_load,tokenizer,curpos_process}.py`、`experiments/hparam_search.py`、
      `data_provider/data_downloader.py`、`training/{metrics,distributed,optimization}.py`、
      `training/callbacks/early_stopping.py`（零引用 re-export；`utils/early_stopping` 改为直接用 `tools.checkpoint.io`）、
      `tests/unit/test_module_moves.py`（守卫测试，垫片全消亡后删除）、
      `references/hf_peft/hf_finetuning/`（仅剩 env.sh 的空壳）、`docs/assets/LLM蒸馏综述.docx`
      （二进制综述文档，经确认无需保留，彻底删除）；`.gitignore` 清除失效的
      `pretraining/examples/gpt2_book/model.pth` 行（全局 `*.pth` 已覆盖）。
- 2026-10-24 `tools/distributed/` 重构：重排为 00–08 学习动线，删除旧 `ddp/`、`dp/`、`fsdp/`、`fairscale/`、
  `tensor_parallelism/` 等空壳与整段外部拷贝；新增数值等价验证与 `tests/unit/test_tools_distributed.py`；详见 [distributed](../70-tools/distributed.md)。
- 2026-10-24 四阶段重构：新增 `training/` 与 `models/base.py`，统一 cache/config/tokenizer/checkpoint 契约，
  预训练/SFT/DPO 共用 `training.Trainer`；当时保留的旧路径 shim 已在本轮全部消亡。
- 2026-10-21 后训练域修复（对照 LLMs-from-scratch ch06/07/附录E）：删除 `sft/data_provider/llmbox/` 死代码树，
  修复 stale import 与 run 入口崩溃、DPO 权重顺序/logprob 口径，`layers/lora.py` 规范化，GRPO completion mask、
  REINFORCE-with-baseline、蒸馏与 judge/MMLU 补全；细节见 [rlhf](../20-post-training/rlhf.md)、[dpo](../20-post-training/dpo.md)。
- 2026-10-20 推断域清理（P1 bug 修复与死文件删除）、合并（P2：judge → judge.py、Chainlit app → app.py、
  启动脚本 → run_app.sh）与通用化（P3：registry/loading/generation 公共 API、nano_vllm 包化）。
- 2026-10-20 layers/models 审查整改（Phase 0–2，见 [layers-models-audit](../60-reference/layers-models-audit/index.md)）：
  11 组 P0 bug 修复、`layers/normalization` 拼写修正、`layers/inference.py` → `inference/generate_core.py`、
  `sys.path` hack 清除、MLA / beam search / ALiBi / SWA / RoPE 外推 / MoE aux loss 等教学主题补全。
- 2026-10-18 预训练域归位与 config 清理：解散 `legacy/` 与 `analysis/`、数据层归位 `data_provider/`、
  config 副作用移除、`max_new_toknes` 全仓拼写修复；详见 [refactor-2026-10-18-domains](../60-reference/refactor-2026-10-18-domains.md)。
- 2026-10-25 输出产物路径规范化：MATH-500 报告 `results/math500/`、checkpoint 统一 `results/checkpoints/`，
  判据与入口默认值见 [outputs](outputs.md)。

## 2. 数据

- `dataset/` 从约 76 GB 清理到约 5.8 GB。
- 删除重复压缩包、HF cache、MMLU auxiliary_train、IMDB 原始目录、DISC-Law QA 子集、CLUENER、通用 SFT 大语料等。
- 保留并归类为 `pretraining/`、`finetuning/`、`evaluation/`；布局与代码路径见 [datasets](datasets.md)。

## 3. 文档

- 文档统一到 `docs/`，按 00–70 编号组织。
- 删除一次性数据集审计页、MMLU 上游 README 副本、旧“现状盘点/路径索引”页。
- `LLMs-from-scratch` 覆盖度分析保留为历史快照，当前架构以 [architecture](../00-overview/architecture.md) 为准。

## 4. 仍存在的遗留

- `post_training/references/trl_sft/run_sft.py` 的 `TrlParser` 用法与当前 `trl` 版本不兼容（`tokenizer=` 已弃用改 `processing_class`）。
- `post_training/peft/demos/scratch_exps/lora_dora_from_scratch.py` 引用不存在的 `data_provider.lora`（四重断链，整体不可运行）。
- `post_training/references/hf_peft/run_peft.py` 是教程拼接（TrainingArguments/Trainer 未导入、数据集未定义），
  会尝试下载 HF 模型，离线/受限环境不可直接运行。
- `post_training/references/` 其余外部栈脚本（trl_grpo / unsloth / hf_peft 各子目录）需按各自依赖与数据单独修复，
  不属于默认主链路。

## 5. 验证

```bash
python -m unittest discover -s tests -v   # 557 项（2026-10-25）
python scripts/check_docs.py              # OK
```
