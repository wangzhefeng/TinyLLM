# 清理记录

> 所属：[TinyLLM Docs](../README.md) › [Development](index.md)
> 更新时间：2026-10-21。本文只记录**当前仍然成立**的清理结果。

## 1. 代码与目录

- 三个主域重组为 `pre_training/`、`post_training/`、`inference/`。
- `post_training/` 拆为 `sft/`、`peft/`、`preference/`、`reasoning/`、`common/`。
- `inference/` 拆为 `serving/`、`api/`、`evaluation/`、`loaders/`、`apps/`、`examples/`。
- 分布式训练独立为 `tools/distributed/`；新增 `tools/profiling/`、`tools/checkpoint/`、`tools/data/`。
- 删除空壳、重复实现和未接入的外部参考，详见 [reference-projects](reference-projects.md)。
- 2026-10-21 后训练域修复（对照 LLMs-from-scratch ch06/07/附录E 与 reasoning-from-scratch）：
    - 删除死代码树 `post_training/sft/data_provider/llmbox/`（LLaMA-Box 残缺复制：
      `Path` 未 import、`process` 签名不匹配、依赖 HF tokenizer 特性、全仓无引用）；
    - 修复全模块 16 处 `utils.llm.*` / `models.gpt` stale import（真实模块在
      `utils/` 平铺与 `models/gpt2/gpt2_124M.py`），新增缺失的 `utils/calc_accuracy.py`；
    - 修复 3 个 run 入口崩溃（`set_seed` 先于 `args_parse`、`args.data` 不存在、
      `re.math` 笔误、`strtobool` 未导入、`required=True` 反模式、GPT-2 config 键名）；
    - 修复 DPO 三处正确性问题：权重加载顺序颠倒（SFT checkpoint 被 HF 权重覆盖）、
      inference 变量互换、logprob 口径改为论文默认的序列求和（可 `--dpo_avg_logprobs 1` 切回平均）；
    - `layers/lora.py`：LoRA 缩放改为标准 `alpha/rank`、支持 dropout 与目标层过滤、
      新增论文版 DoRA；`clf_head.py` 重写为四种清晰冻结策略；
    - `post_training/reasoning/`：GRPO 补 completion mask 与梯度裁剪及多 prompt 批处理、
      新增 REINFORCE-with-baseline（reinforce.py）与推理轨迹蒸馏（distill_reasoning.py）、
      补全 Bradley-Terry / LLM-judge / MMLU first-token logprob 法；
    - `instruction_follow/data_loader.py` 补齐 prompt-masking collate
      （custom_collate_fn_mask_out_instruction，对应书 ch07 exercise）。
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
- 2026-10-20 推断域清理（P1）：
    - 删除死文件：`inference/loaders/qwen3/load_weights_moe.py`（与 load_weights.py 逐行重复且无引用）、
      `inference/scripts/update_codes.sh`（旧仓子模块时代遗物）、`inference/scripts/rm_pyc.sh`（sudo 删除危险且冗余）；
    - 修复 10 处 bug：app_orig 缺 import/None 下标、app_own decode 参数错、openai_evaluate stale import
      （`utils.llm.` → `utils.`）、llama2/llama3 eos id 错误（50256 → 2 / 128001）、llama3 instruct 用错 repo_id、
      llama2/3 下载缓存路径不一致、gemma3 权重误下载到 tokenizer 目录、qwen3 model_download 死分支、
      gpt2_inference 的 `eos_id` kwarg 错误；
    - `openai_gpt2_weights_load.py` 的 tensorflow 改为懒加载（未安装不影响其他加载路径）；
    - `run_app_*.sh` 修复相对路径（脚本内 cd 到 inference/）；
    - 修复断链调用方 `post_training/sft/exp/exp_finetune_gpt_instruction_flow_evaluate.py` 的双 stale import。
- 2026-10-20 推断域合并（P2）：
    - LLM judge 合并：新增 `inference/evaluation/judge.py`（`backend="ollama"|"openai"` 统一入口），
      `ollama_evaluate.py` / `openai_evaluate.py` 改为兼容 shim（原签名不变，内部委托 judge.py）；
    - Chainlit app 合并：新增 `inference/apps/app.py`（`GPT2_WEIGHTS=orig|own` 选择权重来源），
      `app_orig.py` / `app_own.py` 改为兼容 shim（设置环境变量后 import app 注册 handler）；
    - 启动脚本合并：`run_app_orig.sh` / `run_app_own.sh` 合并为 `run_app.sh`（接受 orig|own 参数），旧脚本已删除。
- 2026-10-20 推断域通用化（P3）：
    - 新增公共 API：`inference/registry.py`（MODEL_REGISTRY）、`inference/loading.py`（`load_model` 统一加载）、
      `inference/generation.py`（`generate_text`/`generate_text_stream` 统一生成 facade）、
      `inference/__init__.py`（PEP 562 懒加载导出）；
    - nano_vllm 包化：库模块全部改相对 import、删除 sys.path 操作（不再遮蔽仓根 `config/` 等同名包），
      新增包级 `__init__.py`（导出 LLM/Config/SamplingParams）与 `llm.py` 便捷入口，
      入口脚本保留直跑/-m 双模式；
    - 新增 `tests/unit/test_inference_registry.py`（注册表、load_model 错误路径、facade 离线生成、懒加载契约）。
- 2026-10-20 layers/models 审查整改（Phase 0 修 bug + Phase 1 结构，见
  [layers-models-audit](../60-reference/layers-models-audit/index.md)）：
    - Phase 0 修复 11 组 P0 bug 与低成本 P1（attention 缩放指数、Qwen3-optimized 的 RoPE 张量与
      layer_idx 条件反转、generate 双重索引、pos_embed 回退、embedding assert、FixPE 重复类、
      GPT-2 cache 语义、llama d_ff/eos、Gemma3 权重 tying、tokenizer 系列、MoE 噪声/链式赋值、
      RMSNorm fp32 上转、BPE rank 合并与互逆），新增 `tests/unit/test_layers_regression.py`；
    - `layers/normailzation/` → `layers/normalization/`（拼写修正，全仓 import 同步）；
    - `layers/inference.py` → `inference/generate_core.py`（生成逻辑归位推断域，17 处 import 更新）；
    - 删除重复实现：`models/qwen3/qwen3_original.py`（与 qwen3.py 仅属性名不同）与
      `TransformerBlockQwen3_original`；书本 checkpoint 用
      `inference/loaders/qwen3/load_weights.py:remap_book_state_dict` 重映射加载；
    - 删除 layers/、models/ 全部 37 处 `sys.path.append(Path.cwd())` hack（正式包导入）；
    - `CasualAttention` → `CausalAttention`（保留旧名别名）；llama 模型 `nn.Sequential` → `ModuleList`
      且 embedding 命名 `tok_emb` → `tok_embed`（与 qwen3/gpt2/gemma3 统一，两个 loader 同步）；
    - transformer block 硬编码 `.to(torch.bfloat16)` 改为跟随 `cfg.dtype`；
    - `layers/masking.py` 语义统一为 True=屏蔽（与 attention 一致），新增 `combine_causal_padding_mask`；
    - 修复 `qwen3_batched.py` 的 `pos_id`/`pos_ids` 参数名不匹配（原前向必抛 TypeError）；
    - Qwen3 三模型新增 RoPE 越界防护（超过 context_length 时抛明确 IndexError）。
- 2026-10-20 layers/models 审查整改（Phase 2 补教学主题，见
  [layers-models-audit](../60-reference/layers-models-audit/index.md)）：
    - MLA 完整实现（`models/deepseek/mla.py`）+ `DeepSeekModel` 组装（`model.py`），
      原占位删除；
    - 推断域新增：`beam_search_generate`、`repetition_penalty`（`inference/generate_core.py`）；
    - 预训练域新增：`layers/position_encoding/ALiBi.py`、`SlidingWindowAttention`、
      `trim_kv_cache_window`、RoPE 的 NTK/YaRN 外推、MoE aux loss 与 shared expert；
    - 后训练域新增：`layers/lora.py` 规范 LoRA/DoRA（`post_training/peft/layers/lora_dora.py`
      转为兼容 shim）、GRPO from scratch（`post_training/reasoning/training/reinforcement_learning.py`）、
      KL 蒸馏（`training/distillation.py`）、推断时扩展脚手架
      （`reasoning/inference/voting_and_verification.py` / `advance_text_generation.py`，原占位实现）；
    - 填充 `models/llama/llama31_8B.py`、`llama32_1B.py`（复用 Llama3 Model + 新 config），
      `config/llama/model_cfgs.py` 新增 `LLAMA32_CONFIG_1B`。

- 2026-10-24 `tools/distributed/` 重构（重排为 00–08 学习动线，见 [distributed](../70-tools/distributed.md)）：
    - 删除：整个旧 `ddp/`（含 `DDP-script.py`、`nn_ddp/`）、旧 `dp/`、旧 `fsdp/`
      （含 auto_wrapping/manual_wrapping）、`fairscale/`、旧 `deepspeed_zero/` 空壳、
      `tensor_parallelism/`（含 461 行 torchtitan `llama2_model.py` 拷贝）、`log_utils.py`；
    - 理由：空壳 / 过时 FSDP1 / 整段外部拷贝 / 零引用孤儿 / 名不副实；实质资产已迁入新单元结构；
    - 新增：03/05/06/07 与单进程基线的数值等价验证（全部 PASS）、
      `tests/unit/test_tools_distributed.py`（27 项，慢速层由 `TINYLLM_SLOW_TESTS=1` 启用）；
    - 顺带修复 `utils/ddp_utils.py` 的 `wrap_model`：`device_ids` 由 global_rank 改为 local_rank。

## 2. 数据

- `dataset/` 从约 76 GB 清理到约 5.8 GB。
- 删除重复压缩包、HF cache、MMLU 辅助训练、IMDB 原始目录、DISC-Law QA 子集、CLUENER、通用 SFT 大语料、mobvoi 原始 JSONL 等。
- 保留并归类为 `pretraining/`、`finetuning/`、`evaluation/`；布局与代码路径见 [datasets](datasets.md)。

## 3. 文档

- 文档统一到 `docs/`，按 00–70 编号组织。
- 删除一次性数据集审计页、MMLU 上游 README 副本、旧“现状盘点/路径索引”页。
- `LLMs-from-scratch` 覆盖度分析保留为历史快照，当前架构以 [architecture](../00-overview/architecture.md) 为准。

## 4. 仍存在的遗留

- `post_training/sft/run_sft.py` 的 `TrlParser` 用法与当前 `trl` 版本不兼容（`tokenizer=` 已弃用改 `processing_class`）。
- `post_training/peft/.../lora_dora_from_scratch.py` 引用不存在的 `data_provider.lora`（四重断链，整体不可运行）。
- `post_training/peft/run_peft.py` 是教程拼接（TrainingArguments/Trainer 未导入、数据集未定义），会尝试下载 HF 模型，离线/受限环境不可直接运行。
- `post_training/preference/model_rlhf/ppo/` 仍为空壳（docs 中已如实标注未实现）。
- `post_training/peft/model_peft/prompt_tuning/` 仍为空壳（仅 import 无实现）。

## 5. 验证

```bash
python -m unittest discover -s tests -v   # 85 项（2026-10-20）
python run_gpt2_124M.py --help            # OK
python scripts/check_docs.py              # OK
```

## 6. 2026-10-21 追加（本轮）

- `layers/tokenizers/bpe/tiktoken_bpe.py`：`encode` 透传 `allowed_special`（修端到端暴露的 TypeError）；
- `inference/loaders/gpt2/`：base_config 补 `d_ff`（4×embed_dim）与 `dtype`（layers/feed_forward.py 需要）；
- 空壳补全：`preference/model_rlhf/ppo/main.py`（最小 PPO）、`peft/model_peft/prompt_tuning/main.py`（自研 SoftPromptGPT）、新增 `peft/layers/nf4.py`（NF4 量化原语）；
- 端到端验证：`run_gpt_clf_sft.py` 全链路冒烟通过（130/19/38 批次与书一致，随机权重 30 步 loss 0.699→0.663、valid acc 58.3%>50%、checkpoint 保存 198 keys）；
  HF 权重下载（548MB）在本会话网络下反复 IncompleteRead/停滞（hf-mirror 仅 ~60KB/s），真实权重训练未能在会话内完成——属网络环境问题，代码链路已验证到下载前一步。
