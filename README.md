# TinyLLM

TinyLLM 是一个以 **LLM 预训练 → 后训练 / 逻辑推理 → 模型推断** 为主线的学习与实验仓库。三个任务域共享模型、基础层、配置、训练运行时与评测基础设施。

## 快速索引

| 域 | 代码 | 文档 |
|---|---|---|
| 预训练 | [`pre_training/`](pre_training/) | [docs/10-pretraining/](docs/10-pretraining/index.md) |
| 后训练 / reasoning | [`post_training/`](post_training/) | [docs/20-post-training/](docs/20-post-training/index.md) |
| 模型推断 | [`inference/`](inference/) | [docs/30-inference/](docs/30-inference/index.md) |
| 评测公共库 | [`evaluation/`](evaluation/) | [docs/35-evaluation/](docs/35-evaluation/index.md) |
| 共享训练运行时 | [`training/`](training/) | [docs/00-overview/architecture.md](docs/00-overview/architecture.md) |
| 工具库 | [`tools/`](tools/) | [docs/70-tools/](docs/70-tools/index.md) |
| 共享模型与权重 IO | [`models/`](models/) | [docs/40-models/](docs/40-models/index.md) |
| 共享层/配置/工具 | [`layers/`](layers/)、[`config/`](config/)、[`utils/`](utils/) | [docs/50-development/utils.md](docs/50-development/utils.md) |
| 测试 | [`tests/`](tests/) | [docs/50-development/testing.md](docs/50-development/testing.md) |

**完整文档索引**：[docs/README.md](docs/README.md)

## 快速开始

```bash
# 环境（uv，Python 3.12）
uv sync
source .venv/bin/activate

# 预训练入口
python run_gpt2_124M.py --help
python run_bert.py --help
python run_llama.py --help
python run_qwen.py --help

# 后训练统一入口；任务：sft_clf / sft_ins / dpo / grpo / reinforce / distill_reasoning
python post_training/run.py --task sft_ins --help

# 测试与文档检查
python -m unittest discover -s tests -v
python scripts/check_docs.py
```

Linux + CUDA 12.1 的 PyTorch 安装命令见 `pyproject.toml` 注释。Jupyter 在 dev 依赖组，wandb 在 reference 依赖组。

## 目录概览

```text
TinyLLM/
├── pre_training/        # 预训练：data_provider、experiments、examples(components/training/analysis)、references
├── post_training/       # 后训练：sft、peft、preference、reasoning、rl、references(外部栈对照)
├── inference/           # 推断：loading/generation 门面、harness、strategies、serving、apps、scripts
├── evaluation/          # 评测公共库：clients、verifiers、judges（三域共用）
├── training/            # 共享训练循环、checkpoint、callbacks
├── models/              # 模型定义、registry、models/io 权重 IO（唯一实现）
├── layers/              # attention、block、norm、RoPE、MoE、KV cache、tokenizers、quantization/nf4
├── config/              # Model/Train/Generation 配置
├── tools/               # distributed、profiling、checkpoint、data
├── utils/               # 通用训练与日志工具（含 online.py、early_stopping）
├── dataset/             # 本地数据（gitignored）
├── results/             # 模型与测试输出（gitignored）
├── tests/               # unit / integration
├── docs/                # 全部文档
├── scripts/             # 仓库维护脚本
└── run_*.py             # 预训练薄入口
```

## 架构约定

- 模型、tokenizer 和 checkpoint 有统一契约；GPT-2/Qwen3 使用外部 KV cache 保存生成位置。
- 预训练、SFT、DPO 共用 `training.Trainer`；best 与 resume checkpoint 使用兼容的统一格式。
- 权重加载唯一实现位于 `models/io/`；模型静态元数据统一在 `models/registry.py`（推断加载与预训练实验共用）。
- 评测基础设施上提为公共库 `evaluation/`（客户端/验证器/judge/排行榜）；`inference/harness/` 只保留生成侧评测循环。
- GRPO/PPO/REINFORCE 共享 `post_training/rl/`；`post_training/references/` 是外部栈（TRL/Unsloth/HF peft）对照实验区，不接入 `run.py` 分发器。
- 各主域不再保留兼容 shim；新代码一律使用当前路径，历史清理见 [docs/50-development/cleanup-log.md](docs/50-development/cleanup-log.md)。

## 文档怎么读

采用渐进式披露：根 README 只做索引，各域 `index.md` 提供概要，具体主题拆成小页面。历史分析位于 `docs/60-reference/`。从 [docs/README.md](docs/README.md) 进入完整文档树；术语见 [推断 vs 推理](docs/00-overview/terminology.md)。

## 修改代码时

请先阅读 [AGENTS.md](AGENTS.md)：**功能变更必须同步更新相关文档**。提交前运行完整测试与 `scripts/check_docs.py`；CI 配置位于 `.github/workflows/tests.yml`。
