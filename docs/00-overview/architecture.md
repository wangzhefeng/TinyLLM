> 所属：[TinyLLM Docs](../README.md) › [Overview](index.md)

# TinyLLM 架构与目录边界

> 更新时间：2026-10-24（共享训练运行时与推断归属重构）
> 目标：**预训练 → 后训练 / 逻辑推理 → 模型推断** 三域分离，共享模型协议与训练基础设施。

## 1. 顶层结构

```text
TinyLLM/
├── pre_training/          # 预训练任务与数据
├── post_training/         # SFT / PEFT / preference / reasoning / rl
├── inference/             # generation / strategies / serving / evaluation / api / apps
├── training/              # 共享 Trainer、优化、分布式、checkpoint、callbacks
├── models/                # 模型定义与 models/io 权重 IO
├── layers/                # 共享层、cache、tokenizer、chat template
├── config/、utils/        # typed config 与通用工具
├── tools/                 # 独立工具库
├── dataset/、results/、logs/  # 数据与输出（gitignored）
├── tests/                 # unit / integration
├── docs/、scripts/
└── run_*.py               # 预训练薄入口
```

所有训练和推断产出统一写入 `results/`（结论性产物），过程性输出（服务日志、metrics）写入 `logs/`，判据见 [outputs](../50-development/outputs.md)。旧 `inference/loaders/` 与 reasoning 旧路径只承担兼容转发。

## 2. 三个主域

### 2.1 `pre_training/`

| 目录 | 职责 |
|---|---|
| `data_provider/` | 文本、Gutenberg、BERT 数据；Dataset 保留 CPU tensor |
| `experiments/` | `Model_Pretrain`、BERT 与实验功能 |
| `examples/` | 教材 walkthrough、生成/attention 变体、FLOPs |
| `references/` | 外部参考配置 |
| `scripts/` | 单卡 / DP / DDP 启动 |

GPT-2/Llama/Qwen 入口复用根 parser 和实验流程；BERT 使用独立 encoder 任务流程。分布式初始化、DP/DDP 包装和恢复训练由共享运行时支撑。

### 2.2 `post_training/`

| 目录 | 职责 |
|---|---|
| `sft/` | 分类微调、指令 SFT、BERT 情感分类 |
| `peft/` | LoRA/QLoRA/adapter/prefix/prompt 与参考实验 |
| `preference/` | DPO/PPO 及偏好数据 |
| `reasoning/` | 推理能力训练、蒸馏、教师数据生成（`data_gen.py`）与蒸馏训练入口 |
| `rl/` | GRPO/PPO/REINFORCE 采样、logprob、mask、clipped objective；rewards/tracking/batched、GRPO 变体与训练入口 |
| `common/` | 少量共享数据工具；chat template 已下沉 layers |

统一入口：

```bash
python post_training/run.py --task sft_clf --help
python post_training/run.py --task sft_ins --help
python post_training/run.py --task dpo --help
python post_training/run.py --task grpo --help
python post_training/run.py --task reinforce --help
python post_training/run.py --task distill_reasoning --help
```

### 2.3 `inference/`

| 目录/文件 | 职责 |
|---|---|
| `registry.py`、`loading.py` | 模型注册与统一加载；权重实现调用 `models/io/` |
| `generation.py`、`generate_core.py` | 统一生成 facade 与教材解码 |
| `strategies/` | self-consistency、best-of-n、self-refinement、投机解码 |
| `serving/nano_vllm/` | 独立 serving 优化栈 |
| `api/` | Ollama / OpenAI / OpenRouter 客户端 |
| `evaluation/` | benchmarks（含 advanced parser）、judges、leaderboards、MATH-500 harness（`evaluation/harness/`）与统一 judge |
| `apps/`、`examples/`、`scripts/` | Chainlit UI 与示例 |

批量与流式 facade 均只返回新增文本；GPT-2/Qwen3 默认使用外部 KV cache。`generate_core.py` 中 batched/optimized 变体属于实验后端，不是默认公开链路。

## 3. 共享运行时与资产

| 目录 | 内容 |
|---|---|
| `training/trainer.py` | AMP、梯度累积/裁剪、DDP no_sync、train step、evaluate |
| `training/checkpoint.py` | 统一 model/optimizer/scheduler/epoch/step/config/extra schema |
| `training/distributed.py` | 主链路分布式兼容接口 |
| `training/callbacks/` | early stopping、采样等可选回调 |
| `models/base.py` | CausalLM/EncoderLM 轻量协议 |
| `models/io/` | GPT-2/Qwen3/Llama/Gemma3/Phi-1.5 权重加载与 HF 导出（`export/`）唯一实现 |
| `layers/` | attention、block、FFN、norm、RoPE、MoE、cache、tokenizers |
| `config/` | `ModelConfig`、`TrainConfig`、`GenerationConfig` 与兼容适配 |
| `tools/` | checkpoint、data、profiling、distributed；不依赖三个主域 |
| `utils/` | 通用工具与旧 API 兼容层 |

模型 forward 逐步统一为 `input_ids, cache=None, attention_mask=None, position_ids=None`，旧 `use_cache` 保留兼容。生成位置存于 KV cache，支持同一模型的独立请求交错执行。mask 对外统一 `True=屏蔽`。

## 4. 依赖方向

```text
config / layers / models / tools
              ↓
       training / inference
              ↓
pre_training / post_training / apps / run_*
```

训练与推断均依赖共享模型协议，但彼此不依赖。`models/io` 同时服务训练与推断；`tools` 不引用三个主域，也不依赖主链路 `utils.ddp_utils`。

## 5. 测试

- `tests/unit/`：模块与契约测试，无网络、秒级。
- `tests/integration/`：tiny 预训练+resume、SFT、DPO 与跨模块测试。
- 慢速分布式：`TINYLLM_SLOW_TESTS=1`。
- CI：单元、集成、文档检查；push 时增加 gloo 分布式冒烟。

命令：

```bash
python -m unittest discover -s tests -v
python scripts/check_docs.py
```

历史迁移记录见 [migration-2026-10.md](../60-reference/migration-2026-10.md) 与 [cleanup-log](../50-development/cleanup-log.md)。
