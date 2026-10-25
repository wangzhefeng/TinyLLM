> 所属：[TinyLLM Docs](../README.md) › [Overview](index.md)

# TinyLLM 架构与目录边界

> 更新时间：2026-10-25（评测公共库上提与外部栈 references 分离）
> 目标：**预训练 → 后训练 / 逻辑推理 → 模型推断** 三域分离，共享模型协议、训练运行时与评测基础设施。

## 1. 顶层结构

```text
TinyLLM/
├── pre_training/          # 预训练任务与数据：data_provider / experiments / examples / references / scripts
├── post_training/         # SFT / PEFT / preference / reasoning / rl / references
├── inference/             # 门面 facade + harness / strategies / serving / apps / scripts
├── evaluation/            # 公共评测库：clients / verifiers / judges（三域共用）
├── training/              # 共享 Trainer、checkpoint、callbacks
├── models/                # 模型定义、registry、models/io 权重 IO
├── layers/                # 共享层、cache、tokenizer、chat template、量化原语
├── config/、utils/        # typed config 与通用工具
├── tools/                 # 独立工具库（data / checkpoint / distributed / profiling）
├── dataset/、results/、logs/  # 数据与输出（gitignored）
├── tests/                 # unit / integration
├── docs/、scripts/
└── run_*.py               # 预训练薄入口
```

所有训练和推断产出统一写入 `results/`（结论性产物），过程性输出（服务日志、metrics）写入 `logs/`，判据见 [outputs](../50-development/outputs.md)。

## 2. 三个主域

### 2.1 `pre_training/`

| 目录 | 职责 |
|---|---|
| `data_provider/` | 文本语料 Dataset；`bert/` 数据层（`dapt_data` 经 `tools.data.io.read_text_label_tsv` 读 TSV） |
| `experiments/` | `exp_basic.py`（模型实现经 `models.registry.MODEL_REGISTRY` 解析）与 GPT-2 预训练实验、`bert/dapt_data.py` |
| `examples/` | 教材示例三分组：`components/`（attention/layernorm/FFN/残差/embedding）、`training/`（预训练 demo/训练模板/模型检查）、`analysis/`（FLOPs） |
| `references/` | 外部参考：`gpt2_book/`（书 ch05 standalone）、`pgcorpus/`（Gutenberg 独立训练与数据准备）、`nanotron/` |
| `scripts/` | 单卡 / DP / DDP 启动 |

GPT-2/Llama/Qwen 入口复用根 parser 和实验流程；BERT 使用独立 encoder 任务流程。

### 2.2 `post_training/`

| 目录 | 职责 |
|---|---|
| `sft/` | 分类/指令 SFT、`evaluation/judge.py`（alpaca 指令打分）、BERT 情感分类 |
| `peft/` | `demos/`（lora_demo、prompt_tuning、scratch_exps）、`layers/merge_adapter_weights.py`、QLoRA cfgs |
| `preference/` | DPO（`run/run_gpt_dpo.py`）、RM（`rm/`）、偏好数据 |
| `reasoning/` | 蒸馏训练、教师数据生成（`data_gen.py`） |
| `rl/` | GRPO/PPO/REINFORCE 共享实现（core/objectives/batched/rewards/tracking/grpo_variants/ppo/reinforce）与训练入口 |
| `references/` | **外部栈对照实验区**（TRL/Unsloth/HF peft），保持上游原貌，不接入 `run.py` 分发器 |

统一入口（`run.py` 分发）：

```bash
python post_training/run.py --task sft_clf --help
python post_training/run.py --task sft_ins --help
python post_training/run.py --task dpo --help        # preference/run/run_gpt_dpo.py
python post_training/run.py --task grpo --help
python post_training/run.py --task reinforce --help
python post_training/run.py --task distill_reasoning --help
```

### 2.3 `inference/`

| 目录/文件 | 职责 |
|---|---|
| `loading.py`、`generation.py`、`generate_core.py` | 统一加载 / 生成 facade 与教材解码（顶层门面共 4 文件，含 `__init__.py`） |
| `harness/` | 生成侧评测循环：MATH-500（`math500.py`）与 MMLU（`mmlu.py`） |
| `strategies/` | self-consistency、best-of-n、self-refinement、投机解码 |
| `serving/nano_vllm/` | 独立 serving 优化栈 |
| `apps/`、`examples/`、`scripts/` | Chainlit UI、示例与运维脚本 |

模型注册表在 `models/registry.py`（`inference.loading` 与预训练 `exp_basic` 共用）。批量与流式 facade 均只返回新增文本；GPT-2/Qwen3 默认使用外部 KV cache。`generate_core.py` 中 batched/optimized 变体属于实验后端，不是默认公开链路。

## 3. 公共层

| 目录 | 内容 |
|---|---|
| `training/trainer.py` | AMP、梯度累积/裁剪、DDP no_sync、train step、evaluate |
| `training/checkpoint.py` | 统一 model/optimizer/scheduler/epoch/step/config/extra schema |
| `training/callbacks/` | 采样等可选回调（早停在 `utils/early_stopping.py`，直接用 `tools.checkpoint.io`） |
| `evaluation/` | 三域共用评测基础设施：`clients/`（Ollama/OpenAI/OpenRouter）、`verifiers/`（数学答案验证）、`judges/`（LLM-as-judge + Elo/BT 排行榜），见 [评测公共库](../35-evaluation/index.md) |
| `models/base.py` | CausalLM/EncoderLM 轻量协议 |
| `models/registry.py` | 模型静态元数据注册表（family/config_key/sources/tokenizer） |
| `models/io/` | GPT-2/Qwen3/Llama/Gemma3/Phi-1.5 权重加载与 HF 导出（`export/`）唯一实现 |
| `layers/` | attention、block、FFN、norm、RoPE、MoE、cache、tokenizers、`quantization/nf4.py` |
| `config/` | `ModelConfig`、`TrainConfig`、`GenerationConfig` 与兼容适配 |
| `tools/` | checkpoint、data、profiling、distributed；不依赖三个主域 |
| `utils/` | 通用工具、早停与旧 API 兼容层；`online.py` 提供 HF Hub / wandb 登录 |

模型 forward 统一为 `input_ids, cache=None, attention_mask=None, position_ids=None`。生成位置存于 KV cache，支持同一模型的独立请求交错执行。mask 对外统一 `True=屏蔽`。

## 4. 依赖方向

```text
config / layers / models / tools
              ↓
   training / inference / evaluation
              ↓
pre_training / post_training / apps / run_*
```

训练与推断均依赖共享模型协议，但彼此不依赖；`evaluation/` 同时服务推断 harness 与后训练评测（DPO 数据/judge）。`models/io` 同时服务训练与推断；`tools` 不引用三个主域。`post_training/references/` 与 `pre_training/references/` 是**单向离线参考区**：主链路不 import 它们。

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
