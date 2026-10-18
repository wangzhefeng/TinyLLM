# TinyLLM

TinyLLM 是一个以 **LLM 预训练 → 后训练 / reasoning → 推理** 为主线的学习与实验仓库。代码按三个域组织，模型/层/配置/工具作为共享资产。

## 快速索引

| 域 | 代码 | 文档 |
|---|---|---|
| 预训练 | [`pre_training/`](pre_training/) | [docs/10-pretraining/](docs/10-pretraining/index.md) |
| 后训练 / reasoning | [`post_training/`](post_training/) | [docs/20-post-training/](docs/20-post-training/index.md) |
| 推理 | [`inference/`](inference/) | [docs/30-inference/](docs/30-inference/index.md) |
| 工具库 | [`tools/`](tools/) | [docs/70-tools/](docs/70-tools/index.md) |
| 共享模型 | [`models/`](models/) | [docs/40-models/](docs/40-models/index.md) |
| 共享层/工具 | [`layers/`](layers/)、[`utils/`](utils/)、[`config/`](config/) | [docs/50-development/utils.md](docs/50-development/utils.md) |
| 单元测试 | [`tests/`](tests/) | [docs/50-development/testing.md](docs/50-development/testing.md) |

**完整文档索引**：[docs/README.md](docs/README.md)

## 快速开始

```bash
# 环境（uv，Python 3.12）
uv sync
source .venv/bin/activate

# 预训练主入口
python run_gpt2_124M.py --help

# 单元测试
python -m unittest discover -s tests -v
```

## 目录概览

```text
TinyLLM/
├── pre_training/        # 预训练：data、experiments、examples、references、legacy
├── post_training/       # 后训练：sft、peft、preference、reasoning、common
├── inference/           # 推理：serving、api、evaluation、loaders、apps、examples
├── tools/               # 独立工具库：distributed、profiling、checkpoint、data
├── models/              # gpt2 / llama / qwen3 / gemma3 / bert / deepseek
├── layers/              # attention / block / norm / RoPE / MoE / KV cache / tokenizers
├── config/              # 模型配置
├── utils/               # 训练与日志工具
├── dataset/             # 本地数据（gitignored）
├── tests/               # 单元测试
├── docs/                # 全部文档
├── run_gpt2_124M.py     # 预训练入口
└── AGENTS.md            # 文档/修改规则
```

## 文档怎么读

我们采用**渐进式披露**：

1. 本 README 只做项目总览与入口，不展开细节；
2. 各域/主题入口页（通常为 `index.md`）给概要与目录；
3. 具体主题拆到小页面，页面之间用链接连接；
4. 历史/分析类内容放 `docs/60-reference/`，不再散落在代码目录。

从 [docs/README.md](docs/README.md) 进入完整文档树。

## 当前状态

- 预训练主链路：`run_gpt2_124M.py` + `pre_training/experiments/`
- 单元测试：24 个（`tests/`）
- 历史参考：`docs/60-reference/`（LLMs-from-scratch 分析、迁移记录）
- 数据说明：`docs/50-development/datasets.md`

## 修改代码时

请先阅读 [AGENTS.md](AGENTS.md)：**功能变更必须同步更新相关文档**，并保持“小文档 + 链接”的结构。文档也需要定期人工巡检。
