# LLMs-from-scratch 分析

> 概要、版本快照与逐章/覆盖度/缺口的分页入口。
> **历史快照**：本分析基于 2026-10-02 上游快照与合并前的 TinyLLM；当前架构以 [architecture.md](../../00-overview/architecture.md) 为准，覆盖度结论仅作历史参考。
> **待刷新（2026-10-18）**：附录 D 接入、梯度累积/词表 padding/resume、tokenizer 扩展、内存高效权重加载、Llama/Qwen 入口修复均已完成；matrix.md / gaps.md 的路径口径（`exp/`、`test/` 等）与完成状态已失真，刷新前请以各域页面为准。

> **分析对象**：<https://github.com/rasbt/LLMs-from-scratch>
> **代码快照**：`main` @ `cdbd33e`（2026-10-02），360 个受版本控制文件、68 个 notebook、42 个章节 bonus 子目录
> **对照项目**：`/Users/wangzf/projects/llm_proj/TinyLLM`
> **分析日期**：2026-10-18
> **结论口径**：以“项目主框架内是否有可运行、可复用、与仓库内容等价的实现”为准，不以 README 中列出的链接为准。
>
> **目录说明（2026-10-18 二次整理后）**：本文写作时 `test/llmbox`、`test/minimind`、`apps/`、`distributed_training/` 等仍在主目录。现在仓库已重构为 `pre_training/`、`post_training/`、`inference/` 三个域，并合并了原 `TinyLLMFinetuning`、`TinyLLMInference` 的内容；本文中的旧路径仅作历史参考，当前目录以 [`architecture.md`](../../00-overview/architecture.md) 为准；其中提到的 smolGPT / TinyLlama / MiniMind 参考副本已删除，见 [`reference-projects.md`](../../50-development/reference-projects.md)。

---

## 0. 一页结论

1. **该仓库已经不只是“一本书的配套代码”**。它现在有三层内容：
   - **第一层：书的主线闭环**（ch02–ch07 + 附录 A/D/E），从分词、注意力、GPT、预训练、分类微调一直走到指令微调。
   - **第二层：大量前沿扩展**（42 个 bonus 子目录），包括 KV cache、GQA、MLA、SWA、MoE、DeltaNet、DSA、跨层 KV 共享，以及 Qwen3、Gemma3、OLMo3、Tiny Aya、Qwen3.5、Gemma4 等模型 zoo。
   - **第三层：独立第二本书** `reasoning-from-scratch` 子模块（8 章 + 附录），覆盖推理模型评测、推理时扩展、GRPO、蒸馏，另有可安装的 PyPI 包与 CI。
2. **TinyLLM 的强项在“预训练主线 + 架构横切面”**：GPT-2 预训练闭环、KV cache、GQA、MoE、Qwen3/Gemma3、Llama、DDP/FSDP/TP 等都已具备，部分内容（分布式训练、BERT、Qwen3 30B-A3B、Chainlit）甚至超出原书。
3. **最大的缺口集中在“微调/对齐”这一半本书**：
   - ch06 的 GPT 分类微调（SMS/IMDb）基本没有；
   - ch07 的指令微调（SFT）没有训练闭环；
   - DPO、LoRA（通用 GPT-2 LoRA）、数据集工具、LLM-as-judge 评测、合成数据生成均缺失或只有占位。
4. **架构侧还有 5 个明确的空白**：MLA（只有空类）、独立 SWA、Gated DeltaNet、DSA、跨层 KV 共享；模型 zoo 还缺 OLMo3、Tiny Aya、Qwen3.5、Gemma4。
5. **工程健康度需要先修**：`run_llama.py` / `run_qwen.py` 与 `Model_Pretrain` 签名不匹配，`run_gemma.py` 是空壳，`test/` 下大量脚本仍引用重构前的 `models.gpt2_model_cfg` 路径，`qwen3_reasoning` 的 13 个模块中有 9 个是占位。因此在“补内容”之前，建议先做一次入口与导入路径清理。

**按本报告拆分的 53 个内容单元统计**：

| 状态 | 数量 | 占比 |
|---|---:|---:|
| ✅ 已覆盖（主框架内可运行/等价） | 19 | 36% |
| 🟡 部分覆盖（有零件，未闭环/未集成/旧代码） | 12 | 23% |
| ❌ 未覆盖（无实现或仅有 README 链接） | 20 | 38% |
| ➖ 不适用/无需覆盖 | 2 | 4% |

如果按“代码量 × 完整度”加权，微调/对齐方向的覆盖率会低于 20%，这是当前最值得投入的部分。

---

## 1. 分析方法与口径

### 1.1 分析步骤

1. 浅克隆仓库到 `/tmp/llms-from-scratch`（`git clone --depth 1`），读取 `README.md`、各章 `README.md`、42 个 bonus 子目录的说明与关键脚本。
2. 单独克隆子模块 `reasoning-from-scratch`，确认其章节结构。
3. 对 TinyLLM 做全量文件盘点（排除 `.venv`），核对 `layers/`、`models/`、`exp/`、`data_provider/`、`utils/`、`test/` 中的实现。
4. 对核心模块做导入检查（`.venv/bin/python -c "import ..."`），区分“代码存在”与“代码可运行”。
5. 用 grep 定位 DPO、LoRA、SFT、Ollama、SWA、DSA、DeltaNet、Muon、OLMo、Gemma4、Qwen3.5 等关键词，确认是否有真实实现。
6. 按章节/主题拆成 53 个可评估单元，逐项给出状态与证据路径。

### 1.2 状态定义

| 状态 | 含义 |
|---|---|
| ✅ 已覆盖 | 主框架内有等价实现，能被现有入口调用，或已有验证脚本 |
| 🟡 部分覆盖 | 有组件/数据/旧代码，但缺完整流程、未接入主框架，或存在 stale import |
| ❌ 未覆盖 | 没有实现，或只有空函数/README 链接 |
| ➖ 不适用 | 纯概念章节、硬件差异说明，或与本项目目标无关 |

---

## 2. 仓库全景

### 2.1 顶层结构

| 目录 | 作用 |
|---|---|
| `ch01` | 概念章，无代码；含阅读建议 |
| `ch02` | 文本数据、分词、滑动窗口数据集、DataLoader |
| `ch03` | 自注意力、因果掩码、多头注意力、高效实现 |
| `ch04` | GPT 模型、文本生成、性能分析、KV cache、注意力变体、MoE |
| `ch05` | 预训练闭环、权重加载、Gutenberg、LR scheduler、超参搜索、UI、GPT→Llama、模型 zoo、Muon |
| `ch06` | 分类微调（SMS spam、IMDb） |
| `ch07` | 指令微调、数据集工具、评测、DPO、合成数据、UI |
| `appendix-A` | PyTorch 入门 + DDP |
| `appendix-B` | 参考文献 |
| `appendix-C` | 习题解答索引 |
| `appendix-D` | 训练循环增强：warmup、cosine decay、梯度裁剪 |
| `appendix-E` | LoRA 参数高效微调 |
| `pkg/llms_from_scratch` | 可安装的 PyPI 包（按章组织 API） |
| `setup/` | Python/Docker/SageMaker 环境说明 |
| `reasoning-from-scratch/` | Git 子模块，独立第二本书（推理模型） |

### 2.2 内容规模

- 68 个 notebook：ch05 最多（30 个），ch07 次之（11 个），ch02 有 9 个。
- 42 个章节 bonus 子目录：ch05 占 18 个，ch04 占 9 个，ch02/ch07 各 5 个。
- 仓库的“教学主线”仍然围绕一个 124M 的 GPT-2 小模型；新增内容主要扩展“架构对照”和“后训练”。

### 2.3 三个值得注意的设计特点

1. **每章可独立运行**：主线章节都提供 `previous_chapters.py`、`gpt_download.py` 等自包含依赖，notebook 可以从头跑。
2. **“主线保持简单，bonus 放前沿”**：主章节代码刻意不引入 FlashAttention、分布式、复杂 scheduler；这些放在 bonus，避免破坏教学可读性。
3. **逐步工程化**：`pkg/llms_from_scratch` 把各章代码抽成可安装包，并配 CI；这是从“教学脚本”走向“可复用库”的信号。

### 2.4 仓库的局限

- 主线模型很小（124M），所有训练示例都是“单机、小数据、分钟级”，不等于生产训练。
- bonus 以 notebook/单文件脚本为主，缺少统一抽象；很多实现是“为了对比而复制”，不能直接组合。
- 新模型 zoo 主要验证“前向 + 权重加载 + 生成”，不包含这些模型的预训练/微调流程。
- 评测部分以人工/LLM judge 为主，不是严格 benchmark 框架。

---

## 分页

- [Ch01–03](ch01-03.md)
- [Ch04](ch04.md)
- [Ch05](ch05.md)
- [Ch06–07](ch06-07.md)
- [附录 A–E、pkg、reasoning 子模块](appendices.md)
- [覆盖度矩阵](matrix.md)
- [缺口与落地路线](gaps.md)
