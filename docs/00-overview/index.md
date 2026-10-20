# 项目概览

> 所属：[TinyLLM Docs](../README.md)

TinyLLM 是一个以“从零实现 + 小型实验”为主的 LLM 学习仓库，按模型生命周期拆成三个域。

## 三个域

| 域 | 职责 | 入口 |
|---|---|---|
| 预训练 | 数据、tokenizer、GPT/Llama 等模型、训练循环、分布式参考 | [10-pretraining](../10-pretraining/index.md) |
| 后训练 / reasoning | SFT、PEFT、DPO/RLHF、蒸馏、reasoning-from-scratch | [20-post-training](../20-post-training/index.md) |
| 模型推断 | 权重加载、生成、nano_vllm、评测、UI | [30-inference](../30-inference/index.md) |

## 共享资产

- `models/`：gpt2、llama、qwen3、gemma3、bert、deepseek
- `layers/`：attention、transformer block、FFN、norm、RoPE、MoE、KV cache、tokenizers
- `config/`、`utils/`：配置与训练工具

## 推荐阅读顺序

1. [环境搭建](environment.md)
2. [仓库架构](architecture.md) + [代码→文档映射](repo-map.md)
3. 按兴趣进入三个域
4. 需要背景时看 [LLMs-from-scratch 分析](../60-reference/llms-from-scratch/index.md)

## 相关

- [全部文档索引](../README.md)
- [文档维护规则](../50-development/docs-maintenance.md)
- [根 README](../../README.md)
