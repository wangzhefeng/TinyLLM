> 所属：[TinyLLM Docs](../README.md) › [Overview](index.md)

# 学习路线图

把仓库现有资产按**学习顺序**串成 8 个阶段：不知道从哪开始就按 0 → 7 走，同一阶段内自上而下；只想查某个概念的位置用 [repo-map](repo-map.md)（按目录）。每个知识点给出代码、文档与可运行示例/测试入口。设计参考 [nano-ai.tech 学习路线图](https://nano-ai.tech/map/)（知识点 + 先修关系）与 NamiGPT 的章节式教学。

## 0. 环境与术语

| 知识点 | 入口 |
|---|---|
| 环境搭建（uv、torch、可选依赖） | [environment](environment.md) |
| 推断 vs 推理术语 | [terminology](terminology.md) |
| 仓库架构与目录职责 | [architecture](architecture.md) |

## 1. 推断基础：把 GPT-2 前向吃透

| 知识点 | 代码 | 示例 / 测试 | 文档 |
|---|---|---|---|
| 分词（词级 → BPE） | `layers/tokenizers/` | `examples/components/tokenizer_bpe.py`、`test_tokenizer` | [tokenizers](../40-models/tokenizers.md) |
| 词/位置嵌入 | `layers/embedding/text_embedding.py` | `examples/components/text_embedding.py` | — |
| 位置编码（learned vs 正弦） | `layers/position_encoding/FixPE.py` | `examples/components/pos_encoding.py` | [架构](../40-models/architecture.md) |
| 自注意力 / 因果掩码 / MHA | `layers/attention.py` | `examples/components/attention_walkthrough.py`、`test_attention` | [架构](../40-models/architecture.md) |
| LayerNorm | `layers/normalization/layer_norm.py` | `examples/components/layernorm.py`、`test_norm` | — |
| FFN 与激活函数 | `layers/feed_forward.py` | `examples/components/feed_forward.py`、`activation_gelu.py`、`test_feed_forward` | — |
| Block 与堆叠组装 | `layers/transformer_block.py`、`models/gpt2/` | `examples/components/block_assembly.py`、`test_gpt2` | [GPT-2](../40-models/gpt2.md) |

（示例均在 `pre_training/examples/`，测试在 `tests/unit/`。）

## 2. 生成与采样

| 知识点 | 代码 | 示例 / 测试 | 文档 |
|---|---|---|---|
| 贪心 / 温度 / top-k / top-p | `inference/generate_core.py`、`layers/sampling.py` | `test_decoding_strategies` | [inference](../30-inference/index.md) |
| beam search | `generate_core.beam_search_generate` | `test_decoding_strategies` | [inference](../30-inference/index.md) |
| 每步 top-k/熵追踪（教学） | `inference/interpretability.py` | `test_interpretability` | [可解释性](../30-inference/interpretability.md) |
| 端到端：加载 → 生成 → trace | — | `inference/examples/gpt2_load_generate.py` | [可解释性](../30-inference/interpretability.md) |

## 3. 权重加载与对齐

| 知识点 | 代码 | 示例 / 测试 | 文档 |
|---|---|---|---|
| 三种 GPT-2 加载器（HF/safetensors/TF） | `models/io/gpt2/` | `test_gpt2_loading` | [loaders](../30-inference/loaders.md)、[GPT-2](../40-models/gpt2.md) |
| 与 HF 官方实现逐位对齐 | — | `test_gpt2_hf_alignment` | [GPT-2 对齐](../40-models/gpt2.md) |
| HF 导出（反向） | `models/io/export/` | `test_hf_export` | [loaders](../30-inference/loaders.md) |

## 4. 预训练

| 知识点 | 代码 | 示例 / 测试 | 文档 |
|---|---|---|---|
| 数据与 dataloader | `pre_training/data_provider/` | `test_pretraining`、`test_tools_data` | [data](../10-pretraining/data.md) |
| 训练循环 / 调度 / 裁剪 | `training/` | `examples/training/train_loop_template.py`、`test_training_runtime` | [training](../10-pretraining/training.md) |
| 小数据预训练演示 | — | `examples/training/gpt2_pretrain_demo.py` | [training](../10-pretraining/training.md) |
| resume/checkpoint 契约 | `training/checkpoint.py` | `test_tiny_pretrain_resume`（integration） | [checkpoint](../70-tools/checkpoint.md) |
| 组件升级消融实验 | `pre_training/experiments/exp_component_ablation.py` | 见脚本 | [ablation](../10-pretraining/ablation.md) |

## 5. 后训练

| 知识点 | 代码 | 测试 | 文档 |
|---|---|---|---|
| 指令 SFT / 分类 SFT | `post_training/sft/` | `test_tiny_post_training`（integration） | [finetuning](../20-post-training/finetuning/index.md) |
| LoRA / prompt tuning | `post_training/peft/` | `test_lora`、`test_prompt_tuning` | [peft](../20-post-training/peft.md) |
| DPO / 奖励模型 | `post_training/preference/` | `test_rm` | [dpo](../20-post-training/dpo.md) |
| GRPO / PPO / REINFORCE | `post_training/rl/` | `test_grpo`、`test_ppo`、`test_reinforce` | [rlhf](../20-post-training/rlhf.md) |
| reasoning 训练与蒸馏 | `post_training/reasoning/` | `test_distillation` | [reasoning](../20-post-training/reasoning.md) |

统一入口：`python post_training/run.py --task <task>`。

## 6. 评测

| 知识点 | 代码 | 测试 | 文档 |
|---|---|---|---|
| LLM 客户端 / judge / 验证器 | `evaluation/` | `test_llm_judges`、`test_openai_api` | [评测公共库](../35-evaluation/index.md) |
| 生成侧评测循环（math500 等） | `inference/harness/` | `test_math500_harness` | [evaluation](../30-inference/evaluation.md) |
| 推断时扩展 | `inference/strategies/` | `test_inference_time_scaling` | [scaling](../30-inference/reasoning-scaling.md) |

## 7. 现代架构与服务化

| 知识点 | 代码 | 测试 | 文档 |
|---|---|---|---|
| RoPE（含 scaling） / ALiBi | `layers/position_encoding/` | `test_rope_scaling`、`test_alibi` | [架构](../40-models/architecture.md) |
| GQA / KV cache / 共享 | `layers/attention.py`、`layers/kv_cache.py` | `test_attention`、`test_kv_sharing` | [架构](../40-models/architecture.md) |
| MoE | `layers/moe.py` | `test_moe_eval`、`test_moe_extensions` | [论文映射](../60-reference/paper-map.md) |
| MLA / NSA / MTP（DeepSeek 系） | `models/deepseek/`、`layers/` | `test_mla`、`test_nsa`、`test_mtp` | [deepseek](../40-models/deepseek.md) |
| Qwen3 / Gemma3/4 / Llama | `models/` | `test_qwen3_5`、`test_gemma4` 等 | [models](../40-models/index.md) |
| nano_vllm（分页 KV/TP） | `inference/serving/nano_vllm/` | `test_nano_vllm*` | [nano-vllm](../30-inference/nano-vllm.md) |
| Chainlit UI（含教学模式） | `inference/apps/app.py` | 手动运行 | [serving](../30-inference/serving.md) |

## 相关

- [repo-map](repo-map.md)（按目录查代码→文档）
- [LLMs-from-scratch 分析](../60-reference/llms-from-scratch/index.md)（背景材料）
- [论文→实现映射](../60-reference/paper-map.md)
