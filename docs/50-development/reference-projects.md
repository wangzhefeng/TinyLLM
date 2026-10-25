> 所属：[TinyLLM Docs](../README.md) › [Development](index.md)

# 已移除的外部参考项目

以下项目曾放在 `pre_training/extras/` 与 `post_training/imported/`，但：

- 没有被主代码 import；
- 没有测试或脚本依赖；
- 与当前 LLM 预训练/后训练/推断主线重复或超出范围；
- 都可以从上游重新 clone。

因此从仓库中删除，仅保留链接与用途说明。

| 项目 | 上游 | 原位置 | 原用途 | 删除理由 |
|---|---|---|---|---|
| smolGPT（已删除） | <https://github.com/Om-Alve/smolGPT> | `pre_training/extras/smolGPT/` | 小 GPT 预训练参考：warmup/cosine、梯度累积、`torch.compile`、checkpoint | 主项目已有 `pre_training/examples/train_loop_template.py` 与 `pre_training/experiments/`，需要时重新 clone |
| TinyLlama（已删除） | <https://github.com/jzhang38/TinyLlama> | `pre_training/extras/tinyllama/` | Llama 预训练参考（`llama_model.py` + `pretrain.py`） | 书本主线之外，未接入；需要 Llama 预训练时再引入 |
| MiniMind（已删除） | <https://github.com/jingyaogong/minimind> | `post_training/imported/minimind/` | MiniMind 模型、LoRA、SFT/DPO 数据、tokenizer | 后训练仓库已有 SFT/DPO/PEFT 实现，内容重复 |

## 当前参考项目

- `tools/distributed/`：分布式并行教学参考（DP/DDP/ZeRO/FSDP2/TP/PP/组合 + accelerate/DeepSpeed 对照）；Megatron-LM / torchtitan 上游导读见 [distributed-frameworks](../70-tools/distributed-frameworks.md)。
- `pre_training/references/`：`nanotron/` 配置参考、`gpt2_book/`（LLMs-from-scratch ch05 standalone）、`pgcorpus/`（Gutenberg 独立预训练）。
- `post_training/references/`：外部栈对照实验区——`trl_sft/`、`trl_grpo/`（TRL）、`unsloth/`、`hf_peft/`（HF peft/Transformers 教程）、`lighteval/`（lighteval+vLLM 评测演示，原 TinyLLMInference 快照后新增，2026-10-25 随旧仓库下线抢救迁入）；保持上游原貌，不接入分发器，见 [peft](../20-post-training/peft.md)。

## 说明

- 这些项目的内容没有“丢失”，只是从工作区移除；需要时按上表 URL 重新 clone。
- 如果以后要把它们的某段实现纳入 TinyLLM，请先落到对应域（`pre_training/` 或 `post_training/`）并补测试，而不是整目录复制。
