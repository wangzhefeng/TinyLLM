> 所属：[TinyLLM Docs](../README.md) › [Reference](index.md)

# 论文 → 实现映射

"读论文 → 看本仓库对应实现 → 跑对应测试"的一页索引。按主题分组；arXiv 号附在论文后。图解式论文导读可配合 [nano-ai.tech 论文研读](https://nano-ai.tech/papers/)（Attention 架构图动态版、Switch Transformer 三篇）。

## 基础架构

| 论文 | 实现 | 测试 / 实验 |
|---|---|---|
| Attention Is All You Need（1706.03762） | `layers/attention.py`、`layers/transformer_block.py` | `test_attention` |
| LayerNorm（1607.06450） | `layers/normalization/layer_norm.py` | `test_norm` |
| RMSNorm（1910.07467） | `layers/normalization/rms_norm.py` | `test_norm` |
| GPT-2: Language Models are Unsupervised Multitask Learners | `models/gpt2/gpt2_124M.py` | `test_gpt2`、`test_gpt2_hf_alignment` |
| BERT（1810.04805） | `models/bert/` | `test_bert`、`test_bert_training` |

## 注意力与位置编码

| 论文 | 实现 | 测试 / 实验 |
|---|---|---|
| RoPE（2104.09864） | `layers/position_encoding/RoPE.py` | `test_rope_scaling`、[ablation](../10-pretraining/ablation.md) |
| ALiBi（2108.12409） | `layers/position_encoding/ALiBi.py` | `test_alibi` |
| GQA（2305.13245） | `layers/attention.py::GroupedQueryAttention` | `test_attention`、[ablation](../10-pretraining/ablation.md) |
| MLA（DeepSeek-V2，2405.04434） | `models/deepseek/`、`layers/attention.py` | `test_mla` |
| NSA（2502.11089） | `layers/sparse_attention.py` | `test_nsa` |
| Switch Transformer / MoE（2101.03961） | `layers/moe.py` | `test_moe_eval`、`test_moe_extensions` |

## FFN 与效率

| 论文 | 实现 | 测试 / 实验 |
|---|---|---|
| SwiGLU（2002.05202） | `layers/feed_forward.py::FeedForwardSiLU` | [ablation](../10-pretraining/ablation.md) |
| MTP 多 token 预测（DeepSeek-V3，2412.19437） | `models/deepseek/` | `test_mtp` |
| 投机解码（2211.17192 等） | `inference/strategies/` | `test_decoding_strategies` |

## 参数高效与量化

| 论文 | 实现 | 测试 / 实验 |
|---|---|---|
| LoRA（2106.09685） | `layers/lora.py`、`post_training/peft/` | `test_lora` |
| QLoRA / NF4（2305.14314） | `layers/quantization/nf4.py` | `test_nf4` |

## 对齐与后训练

| 论文 | 实现 | 测试 / 实验 |
|---|---|---|
| InstructGPT / RLHF（2203.02155） | `post_training/preference/`（奖励模型） | `test_rm` |
| DPO（2305.18290） | `post_training/preference/` | `test_tiny_post_training`（integration） |
| PPO（1707.06347） | `post_training/rl/ppo.py` | `test_ppo` |
| GRPO（DeepSeekMath，2402.03300） | `post_training/rl/grpo.py` | `test_grpo` |
| REINFORCE / RFT（2504.11537 等推理型 RL） | `post_training/rl/reinforce.py` | `test_reinforce` |

## 推断与评测

| 论文 | 实现 | 测试 / 实验 |
|---|---|---|
| Self-Consistency（2203.11171） | `inference/strategies/` | `test_inference_time_scaling` |
| Chain-of-Thought（2201.11903） | `inference/harness/math500.py`（CoT 模板） | `test_math500_harness` |
| Bradley-Terry 奖励模型目标（见 RM 综述） | `post_training/preference/`、`evaluation/leaderboards` | `test_rm`、`test_leaderboards` |

## 相关

- [学习路线图](../00-overview/learning-path.md)
- [学习资料链接](learning-resources.md)
- [外部参考项目](../50-development/reference-projects.md)
